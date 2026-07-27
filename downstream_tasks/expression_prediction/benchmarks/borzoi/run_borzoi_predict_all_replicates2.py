#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from borzoi_pytorch import Borzoi
from pyfaidx import Fasta
from tqdm import tqdm


SEQ_LEN = 524_288
MODEL_STRIDE = 32
MODEL_CROP_BINS = 5120
EXPECTED_OUTPUT_BINS = 6144
MIN_BIN_OVERLAP_FRAC = 0.5
DEFAULT_HOME = "/home/jovyan/shares/SR003.nfs2/aspeedok/"

TPM_TO_TARGETS_TRACK = {
    "ENCFF035CWS": "ENCFF387UUZ",
    "ENCFF761SPP": "ENCFF917RKL",
}


@dataclass
class Paths:
    borzoi_dir: Path
    fasta_path: Path
    targets_file: Path
    annot_gtf: Path
    file_mappings_dir: Path


@dataclass
class TargetContext:
    target_index_sub: np.ndarray
    target_scale: np.ndarray
    target_clip_soft: np.ndarray
    target_transform: np.ndarray
    qnorm_target_ids: List[str]
    target_positions_by_gene_strand: Dict[str, List[np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Borzoi/Flashzoi gene-expression prediction for all replicates "
            "and save one CSV per replicate."
        )
    )
    parser.add_argument("model_variant", choices=["borzoi", "flashzoi"])
    parser.add_argument("split", choices=["valid", "test"])
    parser.add_argument(
        "--replicates",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="Replicate ids to run. Default: 0 1 2 3",
    )
    parser.add_argument(
        "--home",
        default=DEFAULT_HOME,
        help="Base home directory used in the original notebook.",
    )
    parser.add_argument(
        "--borzoi-dir",
        default=None,
        help=(
            "Override benchmark directory. If omitted, derived as "
            "$HOME/GENA_LM/downstream_tasks/expression_prediction/benchmarks/borzoi"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where per-replicate prediction CSVs will be saved.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device string, e.g. "cuda:6", "cuda", "cpu". Default: auto',
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug limit on the number of gene rows.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a replicate if its output CSV already exists.",
    )
    parser.add_argument(
        "--no-undo-track-transform",
        action="store_true",
        help="Keep model outputs in squashed scale instead of undoing target transforms.",
    )
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> Paths:
    if args.borzoi_dir:
        borzoi_dir = Path(args.borzoi_dir).expanduser()
    else:
        borzoi_dir = (
            Path(args.home).expanduser()
            / "GENA_LM"
            / "downstream_tasks"
            / "expression_prediction"
            / "benchmarks"
            / "borzoi"
        )

    expr_prediction_dir = borzoi_dir.parent.parent
    datasets_data_dir = expr_prediction_dir / "datasets" / "data"

    return Paths(
        borzoi_dir=borzoi_dir,
        fasta_path=datasets_data_dir / "genomes" / "hg38" / "hg38.fa",
        targets_file=borzoi_dir / "targets_human.txt",
        annot_gtf=borzoi_dir / "gencode.v29.primary_assembly.annotation_UCSC_names.gtf.gz",
        file_mappings_dir=datasets_data_dir / "file_mappings",
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_qnorm_target_id_map(paths: Paths) -> pd.DataFrame:
    expr_map_q = pd.read_csv(paths.file_mappings_dir / "Expression_dataset_v1_csv_file_mappings_qnorm.csv")
    borzoi_map_q = pd.read_csv(paths.file_mappings_dir / "file_mappings_borzoi_human_qnorm.csv")

    qnorm_target_id_map_df = (
        pd.concat([expr_map_q, borzoi_map_q], ignore_index=True)
        .drop_duplicates(subset=["id"], keep="last")[["id", "original_id"]]
        .copy()
    )
    qnorm_target_id_map_df["id"] = qnorm_target_id_map_df["id"].astype(str)
    qnorm_target_id_map_df["original_id"] = qnorm_target_id_map_df["original_id"].astype(str)

    targets_ref = pd.read_csv(paths.targets_file, sep="\t", index_col=0).reset_index(drop=True)
    targets_ref["identifier_base"] = targets_ref["identifier"].astype(str).str.replace(
        r"[+-]$", "", regex=True
    )
    targets_ref["file"] = targets_ref["file"].astype(str)

    def resolve_targets_identifier_base(row: pd.Series) -> object:
        target_id = str(row["id"])
        if target_id in TPM_TO_TARGETS_TRACK:
            return TPM_TO_TARGETS_TRACK[target_id]

        original_id = str(row["original_id"])
        for acc in (original_id, target_id):
            matched = targets_ref.loc[targets_ref["identifier_base"] == acc]
            if len(matched):
                return matched.iloc[0]["identifier_base"]

        for acc in (original_id, target_id):
            matched = targets_ref.loc[targets_ref["file"].str.contains(acc, regex=False, na=False)]
            if len(matched):
                return matched.iloc[0]["identifier_base"]

        return pd.NA

    qnorm_target_id_map_df["targets_identifier_base"] = qnorm_target_id_map_df.apply(
        resolve_targets_identifier_base,
        axis=1,
    )
    return qnorm_target_id_map_df


def load_qnorm_target_id_map(paths: Paths) -> pd.DataFrame:
    map_path = paths.borzoi_dir / "qnorm_id_to_targets_map.csv"
    required_columns = {"id", "original_id", "targets_identifier_base", "strand_specificity"}

    if not map_path.is_file():
        raise FileNotFoundError(f"Required target map is missing: {map_path}")

    cached = pd.read_csv(map_path)
    missing_columns = required_columns.difference(cached.columns)
    if missing_columns:
        raise ValueError(f"Target map is missing required columns {sorted(missing_columns)}: {map_path}")

    cached["id"] = cached["id"].astype(str)
    cached["original_id"] = cached["original_id"].astype(str)
    cached["targets_identifier_base"] = cached["targets_identifier_base"].astype("string")
    cached["strand_specificity"] = cached["strand_specificity"].astype("string").str.strip().str.lower()

    if cached["strand_specificity"].isna().any() or (cached["strand_specificity"] == "").any():
        bad_rows = cached.index[cached["strand_specificity"].isna() | (cached["strand_specificity"] == "")].tolist()
        raise ValueError(f"strand_specificity must be filled for all rows in {map_path}. Bad row indices: {bad_rows[:10]}")

    print(f"Loaded cached qnorm target map: {map_path}")
    return cached


def normalize_strand_specificity(value: object) -> str:
    if pd.isna(value):
        raise ValueError("strand_specificity must be present for every target")

    value_str = str(value).strip().lower()
    if not value_str:
        raise ValueError("strand_specificity must be non-empty for every target")
    return value_str


def choose_track_positions_for_gene_strand(
    track_rows: Dict[str, List[int]],
    gene_strand: str,
    strand_specificity: str,
) -> List[int]:
    del strand_specificity

    plus_rows = track_rows.get("+", [])
    minus_rows = track_rows.get("-", [])
    common_rows = track_rows.get("common", [])

    # No forward-strand swap: plus genes always use plus output rows, and
    # minus genes always use minus output rows. This intentionally ignores
    # strand_specificity for the row-selection rule.
    if gene_strand == "+" and plus_rows:
        return plus_rows
    if gene_strand == "-" and minus_rows:
        return minus_rows
    if common_rows:
        return common_rows
    if plus_rows:
        return plus_rows
    if minus_rows:
        return minus_rows

    raise ValueError(f"No target rows available for gene strand {gene_strand!r}: {track_rows}")


def get_exons_for_transcript_from_gtf_df(
    gtf_df: pd.DataFrame,
    transcript_id: str,
    chrom: Optional[str] = None,
) -> List[tuple[int, int]]:
    mask = (
        gtf_df["feature"].eq("exon")
        & gtf_df["attrs"].astype(str).str.contains(
            fr'transcript_id "{re.escape(transcript_id)}"',
            regex=True,
            na=False,
        )
    )
    exons_df = gtf_df[mask]
    if chrom is not None:
        exons_df = exons_df[exons_df["chrom"] == chrom]
    if exons_df.empty:
        return []

    exons_df = exons_df.sort_values("start")
    return [(int(start) - 1, int(end)) for start, end in exons_df[["start", "end"]].to_numpy()]


def exons_to_bin_ids_min_overlap(
    exons: Iterable[tuple[int, int]],
    out_start: int,
    n_bins: int,
    bin_size: int = MODEL_STRIDE,
    min_frac: float = MIN_BIN_OVERLAP_FRAC,
) -> np.ndarray:
    min_bp = int(np.ceil(bin_size * min_frac))
    out_end = out_start + n_bins * bin_size
    ids: List[int] = []

    for start, end in exons:
        start = max(start, out_start)
        end = min(end, out_end)
        if end <= start:
            continue

        bin0 = max(0, int((start - out_start) // bin_size))
        bin1 = min(n_bins - 1, int((end - 1 - out_start) // bin_size))

        for bin_idx in range(bin0, bin1 + 1):
            bin_start = out_start + bin_idx * bin_size
            bin_end = bin_start + bin_size
            overlap = max(0, min(end, bin_end) - max(start, bin_start))
            if overlap >= min_bp:
                ids.append(bin_idx)

    if not ids:
        return np.array([], dtype=np.int64)
    return np.unique(np.asarray(ids, dtype=np.int64))


def build_target_context(paths: Paths, qnorm_target_id_map_df: pd.DataFrame) -> TargetContext:
    targets_df = pd.read_csv(paths.targets_file, sep="\t", index_col=0).reset_index(drop=True)
    targets_df["target_row"] = np.arange(len(targets_df), dtype=int)
    targets_df["identifier_base"] = targets_df["identifier"].astype(str).str.replace(
        r"[+-]$", "", regex=True
    )

    qnorm_targets_intersection_df = (
        qnorm_target_id_map_df.dropna(subset=["targets_identifier_base"])
        [["id", "original_id", "targets_identifier_base", "strand_specificity"]]
        .drop_duplicates()
        .copy()
    )

    targets_df_sub = targets_df.merge(
        qnorm_targets_intersection_df,
        left_on="identifier_base",
        right_on="targets_identifier_base",
        how="inner",
    ).copy()
    targets_df_sub["track_strand"] = (
        targets_df_sub["identifier"].astype(str).str.extract(r"([+-])$")[0].fillna("common")
    )

    target_index_sub = targets_df_sub["target_row"].to_numpy(dtype=int)
    qnorm_target_ids = targets_df_sub["id"].astype(str).drop_duplicates().tolist()
    strand_specificity_by_id = {
        str(target_id): normalize_strand_specificity(strand_specificity)
        for target_id, strand_specificity in (
            qnorm_targets_intersection_df.drop_duplicates(subset=["id"])[["id", "strand_specificity"]]
            .itertuples(index=False, name=None)
        )
    }

    target_rows_by_id: Dict[str, Dict[str, List[int]]] = {}
    for pos, row in targets_df_sub.reset_index(drop=True).iterrows():
        target_id = str(row["id"])
        track_strand = str(row["track_strand"])
        if target_id not in target_rows_by_id:
            target_rows_by_id[target_id] = {"+": [], "-": [], "common": []}
        target_rows_by_id[target_id].setdefault(track_strand, []).append(pos)

    target_positions_by_gene_strand: Dict[str, List[np.ndarray]] = {"+": [], "-": []}
    for gene_strand in ["+", "-"]:
        for target_id in qnorm_target_ids:
            selected_rows = choose_track_positions_for_gene_strand(
                target_rows_by_id[target_id],
                gene_strand,
                strand_specificity=strand_specificity_by_id.get(target_id),
            )
            target_positions_by_gene_strand[gene_strand].append(
                np.asarray(selected_rows, dtype=int)
            )

    intersection_path = paths.borzoi_dir / "qnorm_targets_human_intersection.csv"
    targets_df_sub[
        [
            "identifier",
            "identifier_base",
            "track_strand",
            "id",
            "original_id",
            "strand_specificity",
            "clip",
            "clip_soft",
            "scale",
            "sum_stat",
            "description",
        ]
    ].to_csv(intersection_path, index=False)
    print(f"Saved target intersection table: {intersection_path}")

    target_sum_stat = targets_df_sub["sum_stat"].astype(str).to_numpy()

    return TargetContext(
        target_index_sub=target_index_sub,
        target_scale=targets_df_sub["scale"].to_numpy(dtype=np.float32),
        target_clip_soft=targets_df_sub["clip_soft"].to_numpy(dtype=np.float32),
        target_transform=np.where(target_sum_stat == "sum_sqrt", 3.0 / 4.0, 1.0).astype(
            np.float32
        ),
        qnorm_target_ids=qnorm_target_ids,
        target_positions_by_gene_strand=target_positions_by_gene_strand,
    )


class Predictor:
    def __init__(
        self,
        *,
        paths: Paths,
        target_context: TargetContext,
        device: torch.device,
        model_variant: str,
        undo_track_transform: bool,
    ) -> None:
        self.paths = paths
        self.target_context = target_context
        self.device = device
        self.model_variant = model_variant
        self.undo_track_transform = undo_track_transform

        self.use_autocast = model_variant == "flashzoi"
        if self.use_autocast and self.device.type != "cuda":
            raise RuntimeError(
                "Flashzoi in upstream Borzoi PyTorch expects a modern Nvidia GPU. "
                "Use --device cuda[:idx]."
            )

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.autocast_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            self.autocast_dtype = None

        self.genome = Fasta(str(paths.fasta_path))
        gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attrs"]
        self.gtf_df = pd.read_csv(paths.annot_gtf, sep="\t", comment="#", names=gtf_cols)
        self.exon_cache: Dict[tuple[str, str], List[tuple[int, int]]] = {}
        self.model: Optional[torch.nn.Module] = None
        self.missing_gene_count = 0

    def load_model(self, replicate: int) -> None:
        model_name = f"johahi/{self.model_variant}-replicate-{replicate}"
        print(f"Loading {model_name} on {self.device}")
        self.model = Borzoi.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def unload_model(self) -> None:
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_autocast_context(self):
        if self.use_autocast and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return nullcontext()

    def one_hot_encode_dna(self, seq: str) -> np.ndarray:
        mapping = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "N": [0, 0, 0, 0],
        }
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for idx, base in enumerate(seq.upper()):
            arr[idx] = mapping.get(base, mapping["N"])
        return arr

    def get_window_sequence(self, row: pd.Series, seq_len: int = SEQ_LEN) -> tuple[str, int]:
        chrom = row["chrom"]
        tss = int(row["TSS"])

        half = seq_len // 2
        start = max(tss - half, 1)
        end = start + seq_len

        seq = self.genome[chrom][start:end].seq
        if len(seq) < seq_len:
            seq = seq + "N" * (seq_len - len(seq))
        elif len(seq) > seq_len:
            seq = seq[:seq_len]

        return seq, start

    def get_transcript_exons(self, transcript_id: str, chrom: str) -> List[tuple[int, int]]:
        key = (transcript_id, chrom)
        if key not in self.exon_cache:
            self.exon_cache[key] = get_exons_for_transcript_from_gtf_df(
                self.gtf_df,
                transcript_id=transcript_id,
                chrom=chrom,
            )
        return self.exon_cache[key]

    def undo_track_transform_from_targets(self, pred_2d: np.ndarray) -> np.ndarray:
        x = pred_2d.astype(np.float32).copy()
        x = x / self.target_context.target_scale[None, :]

        clip_soft = self.target_context.target_clip_soft[None, :]
        mask = x > clip_soft
        x = np.where(mask, (x - clip_soft) ** 2 + clip_soft, x)
        x = x ** (1.0 / self.target_context.target_transform[None, :])
        return x

    def predict_one_sequence(self, one_hot_seq: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        x = torch.from_numpy(one_hot_seq).permute(1, 0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            with self.get_autocast_context():
                y = self.model(x)

        pred = y[0].detach().float().cpu().numpy().transpose(1, 0)
        if pred.shape[0] != EXPECTED_OUTPUT_BINS:
            raise ValueError(
                f"Unexpected number of output bins: {pred.shape[0]} "
                f"(expected {EXPECTED_OUTPUT_BINS})"
            )

        pred = pred[:, self.target_context.target_index_sub]
        if self.undo_track_transform:
            pred = self.undo_track_transform_from_targets(pred)
        return pred

    def collapse_track_expr_to_qnorm_id_expr(
        self,
        expr_by_track: np.ndarray,
        gene_strand: str,
    ) -> np.ndarray:
        gene_strand = str(gene_strand)
        if gene_strand not in self.target_context.target_positions_by_gene_strand:
            raise ValueError(f"Unsupported gene strand: {gene_strand!r}")

        collapsed = np.empty(len(self.target_context.qnorm_target_ids), dtype=np.float32)
        for idx, track_positions in enumerate(
            self.target_context.target_positions_by_gene_strand[gene_strand]
        ):
            collapsed[idx] = float(expr_by_track[track_positions].sum())
        return collapsed

    def gene_vector_for_row(self, row: pd.Series) -> Optional[np.ndarray]:
        seq, seq_start = self.get_window_sequence(row)
        one_hot = self.one_hot_encode_dna(seq)
        pred = self.predict_one_sequence(one_hot)

        transcript_id = str(row.get("transcript_id", "")).strip()
        if not transcript_id or transcript_id.lower() == "nan":
            self.missing_gene_count += 1
            raise ValueError(f"Missing transcript_id for gene: {row['gene_id']}")

        chrom = str(row["chrom"])
        seq_out_start = seq_start + MODEL_STRIDE * MODEL_CROP_BINS
        exons = self.get_transcript_exons(transcript_id, chrom)
        if not exons:
            self.missing_gene_count += 1
            raise ValueError(f"No exons found for transcript_id={transcript_id}")

        exon_bin_ids = exons_to_bin_ids_min_overlap(
            exons,
            out_start=seq_out_start,
            n_bins=pred.shape[0],
            bin_size=MODEL_STRIDE,
            min_frac=MIN_BIN_OVERLAP_FRAC,
        )
        if len(exon_bin_ids) == 0:
            print(f"Transcript has empty exon-bin overlap: {row['gene_id']} / {transcript_id}")
            return None

        exon_pred = pred[exon_bin_ids, :]
        expr_by_track = exon_pred.mean(axis=0)
        expr_by_id = self.collapse_track_expr_to_qnorm_id_expr(
            expr_by_track,
            row["gene_strand"],
        )
        return expr_by_id.astype(np.float32)

    def run_split(
        self,
        *,
        forward_path: Path,
        reverse_path: Path,
        out_csv_path: Path,
        max_rows: Optional[int],
    ) -> pd.DataFrame:
        print(f"\n=== Run split ===\n{forward_path}\n{reverse_path}\n=> {out_csv_path}")

        df_f = pd.read_csv(forward_path, sep="\t")
        df_r = pd.read_csv(reverse_path, sep="\t")
        df = pd.concat([df_f, df_r], ignore_index=True)
        if max_rows is not None:
            df = df.iloc[:max_rows].copy()

        print("Total genes (forward+reverse):", len(df))
        self.missing_gene_count = 0

        gene_to_vecs: Dict[str, List[np.ndarray]] = {}
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                vec = self.gene_vector_for_row(row)
            except Exception as exc:
                print(f"  [WARN] skipping row {idx} ({row['gene_id']}): {exc}")
                continue

            if vec is None:
                continue

            gene_id = str(row["gene_id"])
            gene_to_vecs.setdefault(gene_id, []).append(vec)

        genes: List[str] = []
        preds: List[np.ndarray] = []
        for gene_id, vec_list in gene_to_vecs.items():
            stacked = np.stack(vec_list, axis=0)
            genes.append(gene_id)
            preds.append(stacked.mean(axis=0))

        if not preds:
            raise RuntimeError("No predictions were produced.")

        pred_matrix = np.stack(preds, axis=0)
        final_df = pd.DataFrame(pred_matrix, columns=self.target_context.qnorm_target_ids)
        final_df.insert(0, "gene_id", genes)
        final_df.to_csv(out_csv_path, index=False)

        print("Done:", final_df.shape)
        print("Missing genes count:", self.missing_gene_count)
        return final_df


def main() -> None:
    args = parse_args()
    paths = build_paths(args)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else paths.borzoi_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Model variant: {args.model_variant}")
    print(f"Split: {args.split}")
    print(f"Replicates: {args.replicates}")

    qnorm_target_id_map_df = load_qnorm_target_id_map(paths)
    target_context = build_target_context(paths, qnorm_target_id_map_df)
    predictor = Predictor(
        paths=paths,
        target_context=target_context,
        device=device,
        model_variant=args.model_variant,
        undo_track_transform=not args.no_undo_track_transform,
    )

    forward_path = paths.borzoi_dir / f"human.{args.split}.forward.csv"
    reverse_path = paths.borzoi_dir / f"human.{args.split}.reverse.csv"

    for replicate in args.replicates:
        out_csv_path = output_dir / (
            f"{args.model_variant}_predictions_{args.split}_replicate{replicate}_pytorch.csv"
        )
        if args.skip_existing and out_csv_path.is_file():
            print(f"Skipping replicate {replicate}, output already exists: {out_csv_path}")
            continue

        predictor.load_model(replicate)
        try:
            predictor.run_split(
                forward_path=forward_path,
                reverse_path=reverse_path,
                out_csv_path=out_csv_path,
                max_rows=args.max_rows,
            )
        finally:
            predictor.unload_model()


if __name__ == "__main__":
    main()

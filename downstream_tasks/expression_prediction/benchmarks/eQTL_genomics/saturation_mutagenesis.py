"""Resumable, sharded saturation mutagenesis around expression-model TSSs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import h5py
import numpy as np


DNA = "ACGT"
BASE_TO_CODE = {base: index for index, base in enumerate(DNA)}
CODE_TO_BASE = np.asarray(list(DNA), dtype="U1")
SCHEMA_VERSION = "2"
ORIENTATIONS = (("forward", "+"), ("reverse_complement", "-"))


@dataclass(frozen=True)
class TSSRecord:
    """One validated catalog row with a 0-based TSS coordinate."""

    tss_id: str
    gene_id: str
    gene_name: str
    chromosome: str
    tss_0based: int
    strand: str
    annotation_source: str


@dataclass(frozen=True)
class TSSPlan:
    """Reference sequence and fixed genomic mutation interval for one TSS."""

    record: TSSRecord
    sequence: Any
    center: int
    mutation_start: int
    mutation_end: int
    genomic_start_0based: int
    genomic_end_0based: int
    ref_codes: np.ndarray
    position_offsets: np.ndarray

    @property
    def mutable_count(self) -> int:
        return int(self.ref_codes.size)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alternatives(ref: str) -> tuple[str, str, str]:
    """Return lexicographically ordered alternatives for an A/C/G/T base."""

    if ref not in DNA:
        raise ValueError(f"Reference base must be one of {DNA}, got {ref!r}.")
    return tuple(base for base in DNA if base != ref)  # type: ignore[return-value]


def load_catalog(path: str | Path) -> list[TSSRecord]:
    """Load the documented hg38, 1-based catalog and convert coordinates once."""

    records: list[TSSRecord] = []
    seen: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            tss_id = row["tss_id"]
            if tss_id in seen:
                raise ValueError(f"Duplicate tss_id: {tss_id}")
            seen.add(tss_id)
            position = int(row["tss_position_1based"])
            strand = row["strand"]
            if position < 1 or strand not in {"+", "-"}:
                raise ValueError(f"Invalid catalog row for {tss_id}")
            records.append(
                TSSRecord(
                    tss_id=tss_id,
                    gene_id=row["gene_id"],
                    gene_name=row["gene_name"],
                    chromosome=row["chromosome"],
                    tss_0based=position - 1,
                    strand=strand,
                    annotation_source=row["annotation_source"],
                )
            )
    return records


def render_description(path: str | Path) -> tuple[dict[str, Any], str]:
    """Render JSON with the exact ExpressionDataset training-time formatter."""

    from expression_dataset_final import ExpressionDataset

    description_path = Path(path)
    with description_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    text = ExpressionDataset.make_description_from_json(
        metadata, description_path.stem, str(description_path)
    )
    return metadata, text


def tokenize_description(tokenizer: Any, text: str, text_max_seq_len: int) -> dict[str, Any]:
    """Tokenize exactly like ExpressionDataset.precompute_descriptions."""

    encoded = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=text_max_seq_len,
        return_tensors="pt",
    )
    untruncated = tokenizer(text, padding=False, truncation=False)["input_ids"]
    ids = encoded["input_ids"][0]
    mask = encoded["attention_mask"][0]
    return {
        "description": text,
        "desc_input_ids": ids,
        "desc_attention_mask": mask,
        "token_count": int(ids.numel()),
        "truncated": len(untruncated) > text_max_seq_len,
        "token_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
    }


def _load_checkpoint_runtime_config(path: str | Path) -> dict[str, Any]:
    """Read tokenizer settings saved with the training checkpoint."""

    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(path))
    required = {
        "model_class": "args_params.model_cls",
        "model_input_seq_len": "args_params.input_seq_len",
        "dna_tokenizer": "args_params.gen_tokenizer",
        "description_tokenizer": "args_params.text_tokenizer",
        "text_max_seq_len": "shared_dataset_params.text_max_seq_len",
    }
    values: dict[str, Any] = {}
    for name, key in required.items():
        value = OmegaConf.select(config, key)
        if value is None:
            raise ValueError(f"Checkpoint config {path} is missing required setting {key}")
        values[name] = value
    values["text_max_seq_len"] = int(values["text_max_seq_len"])
    values["model_input_seq_len"] = int(values["model_input_seq_len"])
    if values["text_max_seq_len"] < 1:
        raise ValueError("shared_dataset_params.text_max_seq_len must be positive")
    if values["model_input_seq_len"] < 3:
        raise ValueError("args_params.input_seq_len must allow DNA plus CLS and SEP tokens")
    return values

# TODO: once we reorganize yamls, make yaml file name deterministic based on the checkpoint name
# blocked by @aspeedok
def find_checkpoint_config(checkpoint: str | Path) -> Path:
    """Require exactly one training YAML beside the checkpoint binary."""

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    candidates = sorted(checkpoint_path.parent.glob("*.yaml"))
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise ValueError(
            f"Expected exactly one *.yaml beside {checkpoint_path.name} in "
            f"{checkpoint_path.parent}, found {len(candidates)}: {names}"
        )
    return candidates[0]


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def dna_token_sides(dna_input_seq_len: int, num_before: int) -> tuple[int, int]:
    """Split the DNA-token budget after reserving CLS and SEP tokens."""

    dna_token_budget = int(dna_input_seq_len) - 2
    if not 0 <= int(num_before) <= dna_token_budget:
        raise ValueError(
            f"num_before must be between 0 and {dna_token_budget} for "
            f"DNA input length {dna_input_seq_len}, got {num_before}"
        )
    return int(num_before), dna_token_budget - int(num_before)


def build_provenance(args: Any, rendered: str, desc_tokens: Mapping[str, Any]) -> dict[str, Any]:
    """Build compatibility metadata shared by shards and final outputs."""

    dna_tokens_upstream, dna_tokens_downstream = dna_token_sides(
        args.dna_input_seq_len, args.num_before
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_tss_count": int(args.expected_tss_count),
        "genome_build": "hg38",
        "catalog_sha256": sha256_file(args.catalog),
        "genome_fasta": str(Path(args.genome_fasta).resolve()),
        "genome_fai_sha256": sha256_file(f"{args.genome_fasta}.fai"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_config": str(Path(args.checkpoint_config).resolve()),
        "checkpoint_config_sha256": sha256_file(args.checkpoint_config),
        "model_class": str(args.model_class),
        "model_input_seq_len": int(args.model_input_seq_len),
        "dna_input_seq_len": int(args.dna_input_seq_len),
        "dna_tokenizer": str(args.dna_tokenizer),
        "description_tokenizer": str(args.description_tokenizer),
        "description_json": str(Path(args.description_json).resolve()),
        "description_json_sha256": sha256_file(args.description_json),
        "description": rendered,
        "description_token_count": int(desc_tokens["token_count"]),
        "description_token_ids_sha256": str(desc_tokens["token_ids_sha256"]),
        "text_max_seq_len": int(args.text_max_seq_len),
        "description_truncated": bool(desc_tokens["truncated"]),
        "dna_tokens_upstream": dna_tokens_upstream,
        "dna_tokens_downstream": dna_tokens_downstream,
        "mutation_window_left_bp": 1000,
        "mutation_window_right_bp": 1001,
        "score_window_bp": int(args.score_window_bp),
        "orientations": "forward,reverse_complement",
        "score": (
            f"ATAC channel 0 weighted sum [TSS-{args.score_window_bp},"
            f"TSS+{args.score_window_bp + 1}), alt-ref"
        ),
    }


def create_sequence_plan(
    record: TSSRecord,
    fasta: Any,
    centered_tokenizer: Any,
    *,
    fetch_bp_per_token: int,
) -> TSSPlan:
    """Fetch hg38 context and define the fixed ±1000-bp mutation interval."""

    from gena_expression.sequences import AnnotatedSequence, CoordinateMap, CoordinateSegment, Feature

    if record.chromosome not in fasta.references:
        raise KeyError(f"{record.chromosome} is absent from the hg38 FASTA")
    chrom_length = fasta.get_reference_length(record.chromosome)
    if not 0 <= record.tss_0based < chrom_length:
        raise ValueError(f"TSS outside chromosome for {record.tss_id}")
    fetch = 510 * int(fetch_bp_per_token)
    genomic_start = max(0, record.tss_0based - fetch)
    genomic_end = min(chrom_length, record.tss_0based + fetch)
    sequence_text = fasta.fetch(record.chromosome, genomic_start, genomic_end).upper()
    center = record.tss_0based - genomic_start
    sequence = AnnotatedSequence(
        sequence_text,
        name=record.tss_id,
        features=(Feature("tss", center, center + 1, type="tss", strand=record.strand),),
        coordinate_map=CoordinateMap(
            (
                CoordinateSegment(
                    0,
                    len(sequence_text),
                    source="hg38",
                    chrom=record.chromosome,
                    source_start=genomic_start,
                    source_end=genomic_end,
                    strand="+",
                ),
            )
        ),
        metadata={"genome_build": "hg38", "tss_id": record.tss_id},
    )
    # Validate that both orientations can construct a complete 510+510-token
    # model input, but define variants in base-pair coordinates independently
    # of BPE boundaries.
    for _, orientation_strand in ORIENTATIONS:
        centered_tokenizer.tokenize(sequence, center=center, strand=orientation_strand)
    mutation_genomic_start = max(0, record.tss_0based - 1000)
    mutation_genomic_end = min(chrom_length, record.tss_0based + 1001)
    mutation_start = mutation_genomic_start - genomic_start
    mutation_end = mutation_genomic_end - genomic_start
    reference = sequence_text[mutation_start:mutation_end]
    mutable_offsets = [index for index, base in enumerate(reference) if base in DNA]
    ref_codes = np.asarray([BASE_TO_CODE[reference[index]] for index in mutable_offsets], dtype=np.uint8)
    return TSSPlan(
        record=record,
        sequence=sequence,
        center=center,
        mutation_start=mutation_start,
        mutation_end=mutation_end,
        genomic_start_0based=mutation_genomic_start,
        genomic_end_0based=mutation_genomic_end,
        ref_codes=ref_codes,
        position_offsets=np.asarray(mutable_offsets, dtype=np.int32),
    )


def mutable_offsets(plan: TSSPlan) -> np.ndarray:
    return plan.position_offsets


def mutation_batches(plan: TSSPlan, batch_size: int) -> Iterator[tuple[np.ndarray, list[Any]]]:
    """Lazily materialize alternate sequences and their flattened score indices."""

    offsets = mutable_offsets(plan)
    flat_indices: list[int] = []
    sequences: list[Any] = []
    for base_index, offset in enumerate(offsets):
        local_position = plan.mutation_start + int(offset)
        ref = plan.sequence.sequence[local_position]
        for alt_index, alt in enumerate(alternatives(ref)):
            text = plan.sequence.sequence
            mutated = text[:local_position] + alt + text[local_position + 1 :]
            sequences.append(
                type(plan.sequence)(
                    mutated,
                    name=f"{plan.record.tss_id}:{plan.genomic_start_0based + int(offset) + 1}:{ref}>{alt}",
                    features=plan.sequence.features,
                    coordinate_map=plan.sequence.coordinate_map,
                    metadata=plan.sequence.metadata,
                )
            )
            flat_indices.append(base_index * 3 + alt_index)
            if len(sequences) == batch_size:
                yield np.asarray(flat_indices, dtype=np.int64), sequences
                flat_indices, sequences = [], []
    if sequences:
        yield np.asarray(flat_indices, dtype=np.int64), sequences


def _score_predictions(predictions: Sequence[Any], scorer: Any) -> np.ndarray:
    return np.asarray([scorer.score_prediction(item).score for item in predictions], dtype=np.float32)


def score_plan(
    plan: TSSPlan,
    model: Any,
    condition: Any,
    scorer: Any,
    description_cache: dict[str, dict[str, Any]],
    args: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer absolute reference/mutant ATAC and deltas in both orientations."""

    retention = _prediction_retention()
    reference_scores = np.full(len(ORIENTATIONS), np.nan, dtype=np.float32)
    variant_scores = np.full((plan.mutable_count * 3, len(ORIENTATIONS)), np.nan, dtype=np.float32)
    for orientation_index, (_, orientation_strand) in enumerate(ORIENTATIONS):
        ref_prediction = model.predict_sequence(
            plan.sequence,
            condition=condition,
            center="tss",
            strand=orientation_strand,
            retention=retention,
            description_cache=description_cache,
        )
        reference_scores[orientation_index] = np.float32(
            scorer.score_prediction(ref_prediction).score
        )
        for flat_indices, sequences in mutation_batches(plan, args.batch_size):
            predictions = model.predict_multiple_sequences(
                sequences,
                condition=condition,
                center="tss",
                strand=orientation_strand,
                grouping="condition",
                preprocessing_workers=args.preprocessing_workers,
                preprocessing_backend=args.preprocessing_backend,
                max_records_per_forward=args.batch_size,
                prefetch_batches=args.prefetch_batches,
                show_progress=False,
                retention=retention,
                description_cache=description_cache,
            )
            variant_scores[flat_indices, orientation_index] = _score_predictions(
                predictions, scorer
            )
    deltas = variant_scores - reference_scores.reshape(1, len(ORIENTATIONS))
    return (
        reference_scores,
        variant_scores.reshape(plan.mutable_count, 3, len(ORIENTATIONS)),
        deltas.reshape(plan.mutable_count, 3, len(ORIENTATIONS)),
    )


def _prediction_retention():
    from gena_expression.config import PredictionRetention, RetentionPolicy, ScoringRetention

    return RetentionPolicy(
        prediction=PredictionRetention(
            sequence=True,
            logits=True,
            outputs="scalars",
            tokens=True,
            description_tokens=False,
            provenance=False,
        ),
        scoring=ScoringRetention(prediction=False, tracks="none", features="scores", provenance=False),
    )


def _write_provenance(group: h5py.Group, provenance: Mapping[str, Any]) -> None:
    for key, value in provenance.items():
        group.attrs[key] = value


def initialize_shard(path: Path, plans: Sequence[TSSPlan], provenance: Mapping[str, Any], shard: int, shards: int) -> None:
    """Atomically allocate one resumable shard."""

    if path.exists():
        return
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([plan.mutable_count for plan in plans], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
    with h5py.File(temp, "w") as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["shard_index"] = shard
        output.attrs["shard_count"] = shards
        tss = output.create_group("tss")
        for name, values in {
            "tss_id": [p.record.tss_id for p in plans],
            "gene_id": [p.record.gene_id for p in plans],
            "gene_name": [p.record.gene_name for p in plans],
            "chromosome": [p.record.chromosome for p in plans],
            "strand": [p.record.strand for p in plans],
            "annotation_source": [p.record.annotation_source for p in plans],
        }.items():
            tss.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_string_dtype())
        tss.create_dataset("tss_position_1based", data=[p.record.tss_0based + 1 for p in plans])
        tss.create_dataset("genomic_start_0based", data=[p.genomic_start_0based for p in plans])
        tss.create_dataset("genomic_end_0based", data=[p.genomic_end_0based for p in plans])
        tss.create_dataset("offsets", data=offsets)
        reference = tss.create_dataset(
            "reference_atac_sum",
            data=np.full((len(plans), len(ORIENTATIONS)), np.nan, np.float32),
        )
        reference.attrs["orientation_order"] = ",".join(name for name, _ in ORIENTATIONS)
        tss.create_dataset("status", data=np.zeros(len(plans), np.uint8))
        tss.create_dataset("error", data=np.asarray([""] * len(plans), dtype=object), dtype=_string_dtype())
        ref = output.create_dataset("ref_base", shape=(int(offsets[-1]),), dtype=np.uint8, chunks=True, compression="gzip", compression_opts=1)
        position = output.create_dataset("position_offset", shape=(int(offsets[-1]),), dtype=np.int32, chunks=True, compression="gzip", compression_opts=1)
        chunk_rows = min(32768, max(1, int(offsets[-1])))
        variant = output.create_dataset(
            "variant_atac_sum",
            shape=(int(offsets[-1]), 3, len(ORIENTATIONS)),
            dtype=np.float32,
            chunks=(chunk_rows, 3, len(ORIENTATIONS)),
            compression="gzip",
            compression_opts=1,
            fillvalue=np.nan,
        )
        score = output.create_dataset(
            "scores",
            shape=(int(offsets[-1]), 3, len(ORIENTATIONS)),
            dtype=np.float32,
            chunks=(chunk_rows, 3, len(ORIENTATIONS)),
            compression="gzip",
            compression_opts=1,
            fillvalue=np.nan,
        )
        for index, plan in enumerate(plans):
            ref[offsets[index] : offsets[index + 1]] = plan.ref_codes
            position[offsets[index] : offsets[index + 1]] = plan.position_offsets
        score.attrs["alternative_order"] = "ACGT excluding reference, lexicographic"
        score.attrs["orientation_order"] = ",".join(name for name, _ in ORIENTATIONS)
        variant.attrs["alternative_order"] = score.attrs["alternative_order"]
        variant.attrs["orientation_order"] = score.attrs["orientation_order"]
        _write_provenance(output.create_group("provenance"), provenance)
        output.flush()
    os.replace(temp, path)


def validate_existing_shard(path: Path, plans: Sequence[TSSPlan], provenance: Mapping[str, Any], shard: int, shards: int) -> None:
    with h5py.File(path, "r") as handle:
        if int(handle.attrs["shard_index"]) != shard or int(handle.attrs["shard_count"]) != shards:
            raise ValueError(f"Shard layout mismatch: {path}")
        ids = [item.decode() if isinstance(item, bytes) else str(item) for item in handle["tss/tss_id"][:]]
        if ids != [plan.record.tss_id for plan in plans]:
            raise ValueError(f"TSS assignment mismatch: {path}")
        for key, value in provenance.items():
            if str(handle["provenance"].attrs[key]) != str(value):
                raise ValueError(f"Provenance mismatch for {key}: {path}")


def run_worker(args: Any) -> None:
    """Load one model and process the deterministic TSS subset for a shard."""

    import pysam
    from transformers import AutoTokenizer
    from gena_expression.conditions import Condition
    from gena_expression.inference import SequenceModel
    from gena_expression.scoring import TrackWindowScorer

    args.checkpoint_config = find_checkpoint_config(args.checkpoint)
    checkpoint_runtime = _load_checkpoint_runtime_config(args.checkpoint_config)
    args.model_class = checkpoint_runtime["model_class"]
    args.model_input_seq_len = checkpoint_runtime["model_input_seq_len"]
    args.dna_tokenizer = checkpoint_runtime["dna_tokenizer"]
    args.description_tokenizer = checkpoint_runtime["description_tokenizer"]
    args.text_max_seq_len = checkpoint_runtime["text_max_seq_len"]
    if args.score_window_bp < 0:
        raise ValueError(f"score_window_bp must be non-negative, got {args.score_window_bp}")
    if args.dna_input_seq_len > args.model_input_seq_len:
        raise ValueError(
            f"DNA input length {args.dna_input_seq_len} exceeds checkpoint model "
            f"capacity {args.model_input_seq_len}"
        )
    dna_token_sides(args.dna_input_seq_len, args.num_before)
    records = load_catalog(args.catalog)
    if args.limit is not None:
        records = records[: args.limit]
    args.expected_tss_count = len(records)
    records = [record for index, record in enumerate(records) if index % args.shard_count == args.shard_index]
    dna_tokenizer = AutoTokenizer.from_pretrained(str(args.dna_tokenizer))
    desc_tokenizer = AutoTokenizer.from_pretrained(str(args.description_tokenizer), padding_side="left")
    _, rendered = render_description(args.description_json)
    desc_tokens = tokenize_description(desc_tokenizer, rendered, args.text_max_seq_len)
    provenance = build_provenance(args, rendered, desc_tokens)
    fasta = pysam.FastaFile(str(args.genome_fasta))

    # A light tokenizer object is sufficient to pre-plan exact reference spans.
    from gena_expression.inference.tokenization import CenteredTokenizer
    planner = CenteredTokenizer(
        dna_tokenizer,
        args.dna_input_seq_len,
        args.fetch_bp_per_token,
        args.num_before,
    )
    plans = [create_sequence_plan(record, fasta, planner, fetch_bp_per_token=args.fetch_bp_per_token) for record in records]
    shard_path = Path(args.output_dir) / f"shard-{args.shard_index:04d}-of-{args.shard_count:04d}.h5"
    initialize_shard(shard_path, plans, provenance, args.shard_index, args.shard_count)
    validate_existing_shard(shard_path, plans, provenance, args.shard_index, args.shard_count)

    os.environ.setdefault("CAGI5_MODEL_ROOT", str(Path(args.checkpoint).resolve().parents[1]))
    model = SequenceModel.load(
        model_cls=str(args.model_class),
        checkpoint=args.checkpoint,
        config=args.checkpoint_config,
        dna_tokenizer=args.dna_tokenizer,
        description_tokenizer=args.description_tokenizer,
        dna_max_seq_len=args.dna_input_seq_len,
        desc_max_seq_len=args.text_max_seq_len,
        token_len_for_fetch=args.fetch_bp_per_token,
        num_before=args.num_before,
        device=args.device,
    )
    condition = Condition(name=Path(args.description_json).stem, description=rendered)
    scorer = TrackWindowScorer(
        track="atac",
        center="tss",
        left_bp=args.score_window_bp,
        right_bp=args.score_window_bp + 1,
        aggregate="sum",
        channel=0,
    )
    description_cache = {model._grouping_key_for_condition(condition): {
        "description": rendered,
        "desc_input_ids": desc_tokens["desc_input_ids"],
        "desc_attention_mask": desc_tokens["desc_attention_mask"],
    }}
    started = time.monotonic()
    completed_variants = 0
    with h5py.File(shard_path, "r+") as output:
        offsets = output["tss/offsets"][:]
        for index, plan in enumerate(plans):
            if output["tss/status"][index] == 1:
                continue
            try:
                reference_scores, variant_scores, scores = score_plan(
                    plan, model, condition, scorer, description_cache, args
                )
                start, end = int(offsets[index]), int(offsets[index + 1])
                output["variant_atac_sum"][start:end, :, :] = variant_scores
                output["scores"][start:end, :] = scores
                output["tss/reference_atac_sum"][index, :] = reference_scores
                output["tss/error"][index] = ""
                output["tss/status"][index] = 1
                completed_variants += plan.mutable_count * 3
            except Exception as exc:
                output["tss/status"][index] = 2
                output["tss/error"][index] = f"{type(exc).__name__}: {exc}"
                output.flush()
                if args.on_error == "raise":
                    raise
            output.flush()
            elapsed = max(time.monotonic() - started, 1e-9)
            print(f"shard={args.shard_index} tss={index + 1}/{len(plans)} variants_per_second={completed_variants / elapsed:.2f}", flush=True)


def _copy_rows(source: h5py.File, target: h5py.File, tss_at: int, base_at: int) -> tuple[int, int]:
    n_tss = len(source["tss/tss_id"])
    n_bases = len(source["ref_base"])
    for name in source["tss"]:
        if name == "offsets":
            continue
        target[f"tss/{name}"][tss_at : tss_at + n_tss] = source[f"tss/{name}"][:]
    target["ref_base"][base_at : base_at + n_bases] = source["ref_base"][:]
    target["position_offset"][base_at : base_at + n_bases] = source["position_offset"][:]
    target["variant_atac_sum"][base_at : base_at + n_bases] = source["variant_atac_sum"][:]
    target["scores"][base_at : base_at + n_bases] = source["scores"][:]
    return tss_at + n_tss, base_at + n_bases


def merge_shards(paths: Sequence[Path], output_path: Path) -> None:
    """Validate complete compatible shards and atomically create one final HDF5."""

    if not paths:
        raise ValueError("No shard files were found.")
    handles = [h5py.File(path, "r") for path in paths]
    try:
        expected_count = int(handles[0].attrs["shard_count"])
        indices = [int(handle.attrs["shard_index"]) for handle in handles]
        if len(handles) != expected_count or sorted(indices) != list(range(expected_count)):
            raise ValueError(f"Expected shards 0..{expected_count - 1}, found {sorted(indices)}")
        provenance = dict(handles[0]["provenance"].attrs)
        ids: list[str] = []
        for handle in handles:
            if dict(handle["provenance"].attrs) != provenance:
                raise ValueError("Shard provenance differs.")
            if not np.all(handle["tss/status"][:] == 1):
                raise ValueError(f"Incomplete shard: {handle.filename}")
            ids.extend(item.decode() if isinstance(item, bytes) else str(item) for item in handle["tss/tss_id"][:])
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate TSS IDs across shards.")
        total_tss = sum(len(handle["tss/tss_id"]) for handle in handles)
        expected_tss = int(provenance["expected_tss_count"])
        if total_tss != expected_tss:
            raise ValueError(f"Expected {expected_tss} TSS records, found {total_tss}")
        total_bases = sum(len(handle["ref_base"]) for handle in handles)
        temp = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temp, "w") as target:
            target.attrs["schema_version"] = SCHEMA_VERSION
            tss = target.create_group("tss")
            first = handles[0]
            for name, dataset in first["tss"].items():
                if name == "offsets":
                    continue
                dtype = _string_dtype() if h5py.check_string_dtype(dataset.dtype) else dataset.dtype
                tss.create_dataset(
                    name,
                    shape=(total_tss, *dataset.shape[1:]),
                    dtype=dtype,
                )
            target.create_dataset("ref_base", shape=(total_bases,), dtype=np.uint8, chunks=True, compression="gzip", compression_opts=1)
            target.create_dataset("position_offset", shape=(total_bases,), dtype=np.int32, chunks=True, compression="gzip", compression_opts=1)
            chunk_rows = min(32768, max(1, total_bases))
            target.create_dataset("variant_atac_sum", shape=(total_bases, 3, len(ORIENTATIONS)), dtype=np.float32, chunks=(chunk_rows, 3, len(ORIENTATIONS)), compression="gzip", compression_opts=1)
            target.create_dataset("scores", shape=(total_bases, 3, len(ORIENTATIONS)), dtype=np.float32, chunks=(chunk_rows, 3, len(ORIENTATIONS)), compression="gzip", compression_opts=1)
            target["variant_atac_sum"].attrs["alternative_order"] = "ACGT excluding reference, lexicographic"
            target["variant_atac_sum"].attrs["orientation_order"] = ",".join(name for name, _ in ORIENTATIONS)
            target["scores"].attrs.update(target["variant_atac_sum"].attrs)
            tss_at = base_at = 0
            counts: list[int] = []
            for source in handles:
                source_offsets = source["tss/offsets"][:]
                counts.extend(np.diff(source_offsets).tolist())
                tss_at, base_at = _copy_rows(source, target, tss_at, base_at)
            tss.create_dataset("offsets", data=np.concatenate(([0], np.cumsum(counts))).astype(np.int64))
            _write_provenance(target.create_group("provenance"), provenance)
            target.flush()
        os.replace(temp, output_path)
    finally:
        for handle in handles:
            handle.close()


def iter_variant_scores(path: str | Path, tss_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield explicit 1-based genomic SNV records from the compact HDF5."""

    with h5py.File(path, "r") as handle:
        ids = [item.decode() if isinstance(item, bytes) else str(item) for item in handle["tss/tss_id"][:]]
        offsets = handle["tss/offsets"][:]
        for index, current_id in enumerate(ids):
            if tss_id is not None and current_id != tss_id:
                continue
            start, end = int(offsets[index]), int(offsets[index + 1])
            genomic_start = int(handle["tss/genomic_start_0based"][index])
            strand_raw = handle["tss/strand"][index]
            strand = strand_raw.decode() if isinstance(strand_raw, bytes) else str(strand_raw)
            codes = handle["ref_base"][start:end]
            positions = handle["position_offset"][start:end]
            scores = handle["scores"][start:end]
            variant_values = handle["variant_atac_sum"][start:end]
            reference_values = handle["tss/reference_atac_sum"][index]
            for position_offset, code, values, absolute_values in zip(
                positions, codes, scores, variant_values
            ):
                ref = DNA[int(code)]
                for alt, orientation_deltas, orientation_absolutes in zip(
                    alternatives(ref), values, absolute_values
                ):
                    yield {
                        "tss_id": current_id,
                        "chromosome": (handle["tss/chromosome"][index].decode() if isinstance(handle["tss/chromosome"][index], bytes) else str(handle["tss/chromosome"][index])),
                        "position_1based": genomic_start + int(position_offset) + 1,
                        "ref": ref,
                        "alt": alt,
                        "forward_reference_atac_sum": float(reference_values[0]),
                        "forward_variant_atac_sum": float(orientation_absolutes[0]),
                        "forward_delta": float(orientation_deltas[0]),
                        "reverse_complement_reference_atac_sum": float(reference_values[1]),
                        "reverse_complement_variant_atac_sum": float(orientation_absolutes[1]),
                        "reverse_complement_delta": float(orientation_deltas[1]),
                    }

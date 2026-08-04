#!/usr/bin/env python3
"""Reproduce the SOX2 ISM notebook with its exact sequence and joint-pair workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
API_SRC = TASK_DIR / "api" / "src"
for path in (REPO_ROOT, TASK_DIR, API_SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from saturation_mutagenesis import (  # noqa: E402
    _load_checkpoint_runtime_config,
    find_checkpoint_config,
    sha256_file,
)


SOX2_CHROM = "chr3"
SOX2_TSS_0BASED = 181_711_924
SOX2_TES_0BASED_EXCLUSIVE = 181_714_436
NUM_BEFORE = 511
TOKEN_LEN_FOR_FETCH = 20
DNA_INPUT_SEQ_LEN = 1024
PROMOTER_BP = 600
VARIANT_SCORE_WIDTH_BP = 501


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "models" / "dev_loss" / "pytorch_model.bin",
    )
    parser.add_argument(
        "--genome-fasta",
        type=Path,
        default=REPO_ROOT / "data" / "genomes" / "hg38" / "hg38.fa",
    )
    parser.add_argument(
        "--description-json",
        type=Path,
        default=TASK_DIR / "api" / "data" / "real_descriptions" / "ENCFF081FQX.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preprocessing-workers", type=int, default=10)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--max-pairs-per-forward", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=1600)
    parser.add_argument(
        "--include-deletions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the notebook's 20 sampled single-base deletions.",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "sox2_notebook_reproduction.parquet"
    plot_path = args.output_dir / "sox2_notebook_reproduction_track_window.png"
    provenance_path = args.output_dir / "sox2_notebook_reproduction_provenance.json"
    for path in (parquet_path, plot_path, provenance_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    from gena_expression.conditions import DescriptionLookup, metadata_to_description
    from gena_expression.dna import Genome
    from gena_expression.inference import SequenceModel
    from gena_expression.ism import ISM
    from gena_expression.scoring import TrackWindowScorer, VariantInterpreter

    checkpoint_config = find_checkpoint_config(args.checkpoint)
    runtime = _load_checkpoint_runtime_config(checkpoint_config)
    os.environ.setdefault("CAGI5_MODEL_ROOT", str(args.checkpoint.resolve().parents[1]))
    model = SequenceModel.load(
        model_cls=str(runtime["model_class"]),
        checkpoint=args.checkpoint,
        config=checkpoint_config,
        dna_tokenizer=runtime["dna_tokenizer"],
        description_tokenizer=runtime["description_tokenizer"],
        dna_max_seq_len=DNA_INPUT_SEQ_LEN,
        num_before=NUM_BEFORE,
        desc_max_seq_len=int(runtime["text_max_seq_len"]),
        token_len_for_fetch=TOKEN_LEN_FOR_FETCH,
        device=args.device,
    )

    genome = Genome(fasta_path=args.genome_fasta)
    sequence_start = SOX2_TSS_0BASED - NUM_BEFORE * TOKEN_LEN_FOR_FETCH
    center = SOX2_TSS_0BASED - sequence_start
    sequence = genome.sequence(
        chrom=SOX2_CHROM,
        start=sequence_start,
        end=SOX2_TES_0BASED_EXCLUSIVE,
        strand="+",
        name="SOX2 gene region",
    ).add_feature(
        name="promoter",
        start=center - PROMOTER_BP,
        end=center,
        type="promoter",
    )

    lookup = DescriptionLookup(
        json_dir=args.description_json.parent,
        keys=["H1"],
        H1={"id": "ENCFF081FQX"},
    )
    condition_metadata = lookup["H1"]
    ism = ISM(
        sequence=sequence,
        center=center,
        region="promoter",
        condition=condition_metadata,
        n_deletions=20 if args.include_deletions else 0,
        n_substitutions=None,
    )
    scorer = TrackWindowScorer(center="variant", width_bp=VARIANT_SCORE_WIDTH_BP)
    started = time.monotonic()
    ism.score(
        interpreter=VariantInterpreter(model),
        scorer=scorer,
        pair_execution="joint",
        preprocessing_workers=args.preprocessing_workers,
        prefetch_batches=args.prefetch_batches,
        max_pairs_per_forward=args.max_pairs_per_forward,
        chunk_size=args.chunk_size,
    )
    elapsed = time.monotonic() - started
    ism.write_parquet(parquet_path)
    figure, _ = ism.plot(scorer_name="track_window", mutation_type="all")
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")

    provenance = {
        "experiment": "sox2_dev_loss_h1_notebook_exact",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_md5": md5_file(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_config": str(checkpoint_config),
        "checkpoint_config_sha256": sha256_file(checkpoint_config),
        "genome_fasta": str(args.genome_fasta.resolve()),
        "description_json": str(args.description_json.resolve()),
        "description_json_sha256": sha256_file(args.description_json),
        "rendered_description": metadata_to_description(condition_metadata),
        "chromosome": SOX2_CHROM,
        "tss_0based": SOX2_TSS_0BASED,
        "tss_1based": SOX2_TSS_0BASED + 1,
        "tes_0based_exclusive": SOX2_TES_0BASED_EXCLUSIVE,
        "sequence_start_0based": sequence_start,
        "sequence_end_0based": SOX2_TES_0BASED_EXCLUSIVE,
        "sequence_length_bp": len(sequence),
        "promoter_start_local": center - PROMOTER_BP,
        "promoter_end_local": center,
        "dna_input_seq_len": DNA_INPUT_SEQ_LEN,
        "num_before": NUM_BEFORE,
        "token_len_for_fetch": TOKEN_LEN_FOR_FETCH,
        "score": "ATAC channel 0 weighted sum, variant-centered width 501 bp, alt-ref",
        "pair_execution": "joint",
        "substitutions": 1800,
        "deletions": 20 if args.include_deletions else 0,
        "elapsed_seconds": elapsed,
        "variants_per_second": len(ism.variants) / elapsed,
    }
    write_json(provenance_path, provenance)
    print(
        f"variants={len(ism.variants)} elapsed_seconds={elapsed:.2f} "
        f"variants_per_second={len(ism.variants) / elapsed:.2f}",
        flush=True,
    )
    print(f"wrote {parquet_path}\nwrote {plot_path}\nwrote {provenance_path}")


if __name__ == "__main__":
    main()

"""Stable result-table, metadata, and FASTA exports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from gena_expression import AnnotatedSequence, SequenceOptimizer

from .config import RunConfig, json_ready


def csv_safe(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert lineage lists to text because CSV has no list data type."""

    if "parent_ids" not in frame.columns:
        return frame
    return frame.with_columns(pl.col("parent_ids").list.join(";").alias("parent_ids"))


def write_table_pair(frame: pl.DataFrame, stem: Path) -> list[Path]:
    """Write one table as type-preserving Parquet and convenient CSV."""

    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    frame.write_parquet(parquet_path)
    csv_safe(frame).write_csv(csv_path)
    return [parquet_path, csv_path]


def fasta_lines(records: Iterable[tuple[str, str]]) -> Iterable[str]:
    """Yield sanitized FASTA records with sequence lines wrapped at 80 bases."""

    for header, sequence in records:
        clean_header = "_".join(str(header).split())
        yield f">{clean_header}\n"
        for start in range(0, len(sequence), 80):
            yield f"{sequence[start:start + 80]}\n"


def write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> Path:
    """Write named DNA strings to one FASTA file."""

    path.write_text("".join(fasta_lines(records)), encoding="utf-8")
    return path


def generation_min_on_max_off(history: pl.DataFrame) -> pl.DataFrame:
    """Return the compact post-selection generation table from the original runner."""

    return history.select(
        pl.col("generation"),
        pl.col("best_min_on_target").alias("min_on_target"),
        pl.col("best_max_off_target").alias("max_off_target"),
    )


def _write_json(path: Path, value: Any) -> Path:
    """Write one deterministic, human-readable JSON document."""

    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _sample_metadata(samples: Sequence[AnnotatedSequence]) -> list[dict[str, Any]]:
    """Return auditable source metadata for the genomic starting population."""

    return [
        {
            "sample_index": index,
            "optimizer_role": "reference" if index == 0 else "initial",
            "name": sample.name,
            "sequence_length": len(sample),
            "metadata": dict(sample.metadata),
        }
        for index, sample in enumerate(samples)
    ]


def write_results(
    optimizer: SequenceOptimizer,
    output_dir: Path,
    *,
    config: RunConfig,
    sampled_population: Sequence[AnnotatedSequence],
) -> list[Path]:
    """Write the original result contract plus sampled-population provenance."""

    written: list[Path] = []
    for table_name in ("evaluations", "final_population", "history"):
        written.extend(write_table_pair(getattr(optimizer, table_name), output_dir / table_name))

    compact_history = generation_min_on_max_off(optimizer.history)
    written.extend(write_table_pair(compact_history, output_dir / "generation_min_on_max_off"))

    written.append(_write_json(output_dir / "summary.json", optimizer.summary()))
    written.append(_write_json(output_dir / "run_config.json", config.as_dict()))

    evaluated_regions_path = output_dir / "evaluated_region_sequences.fasta"
    write_fasta(
        evaluated_regions_path,
        (
            (
                f"{row['candidate_id']}|generation={row['generation_born']}|fitness={row['fitness']}",
                row["region_sequence"],
            )
            for row in optimizer.evaluations.iter_rows(named=True)
        ),
    )
    written.append(evaluated_regions_path)

    final_regions_path = output_dir / "final_population_region_sequences.fasta"
    write_fasta(
        final_regions_path,
        (
            (
                f"{row['candidate_id']}|rank={row['rank']}|fitness={row['fitness']}",
                row["region_sequence"],
            )
            for row in optimizer.final_population.iter_rows(named=True)
        ),
    )
    written.append(final_regions_path)

    final_full_path = output_dir / "final_population_full_sequences.fasta"
    write_fasta(
        final_full_path,
        (
            (
                f"{row['candidate_id']}|rank={row['rank']}|fitness={row['fitness']}",
                row["sequence"],
            )
            for row in optimizer.final_population.iter_rows(named=True)
        ),
    )
    written.append(final_full_path)

    best_row = optimizer.final_population.row(0, named=True)
    best_path = output_dir / "best_full_sequence.fasta"
    write_fasta(
        best_path,
        [
            (
                f"{best_row['candidate_id']}|rank=1|fitness={best_row['fitness']}",
                best_row["sequence"],
            )
        ],
    )
    written.append(best_path)

    initial_path = output_dir / "initial_population.fasta"
    write_fasta(
        initial_path,
        (
            (
                f"sample_{index:04d}|role={'reference' if index == 0 else 'initial'}|source={sample.name or 'unnamed'}",
                sample.sequence,
            )
            for index, sample in enumerate(sampled_population)
        ),
    )
    written.append(initial_path)
    written.append(
        _write_json(
            output_dir / "initial_population_metadata.json",
            _sample_metadata(sampled_population),
        )
    )

    region_length = optimizer.region.end - optimizer.region.start
    manifest_path = output_dir / "output_manifest.json"
    manifest = {
        "files": [path.name for path in written] + [manifest_path.name],
        "generation_min_on_max_off_semantics": {
            "population": "selected population after selection in each generation",
            "min_on_target": "highest candidate-level minimum on-target score in the selected population",
            "max_off_target": "lowest candidate-level maximum off-target score in the selected population",
            "same_candidate_required": False,
        },
        "sequence_exports": {
            "region_length": region_length,
            "evaluated_region_sequences.fasta": "all unique scored candidate regions",
            "final_population_region_sequences.fasta": "selected final candidate regions",
            "final_population_full_sequences.fasta": "selected final candidates in plasmid context",
            "best_full_sequence.fasta": "rank-one candidate in plasmid context",
            "initial_population.fasta": "genome-sampled starting promoters; sample zero is pasted into the plasmid",
        },
    }
    written.append(_write_json(manifest_path, manifest))
    return written

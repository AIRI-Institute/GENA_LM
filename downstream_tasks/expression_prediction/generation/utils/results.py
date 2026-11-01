"""Stable result-table, metadata, and FASTA exports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from gena_expression import AnnotatedSequence, SequenceOptimizer

from .config import RunConfig, json_ready


def csv_safe(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert lineage lists to text because CSV has no list data type."""

    if "parent_ids" not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col("parent_ids")
        .cast(pl.List(pl.String))
        .list.join(";")
        .alias("parent_ids")
    )


def write_table_pair(frame: pl.DataFrame, stem: Path) -> list[Path]:
    """Write one table as type-preserving Parquet and convenient CSV."""

    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    frame.write_parquet(parquet_path)
    csv_safe(frame).write_csv(csv_path)
    return [parquet_path, csv_path]


def insertion_only_table(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove a materialized plasmid column while preserving insertion data."""

    return frame.drop("sequence") if "sequence" in frame.columns else frame


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
    """Return each generation's best-fitness candidate and target metrics."""

    return history.select(
        pl.col("generation"),
        pl.col("best_candidate_id").alias("candidate_id"),
        pl.col("best_fitness").alias("fitness"),
        pl.col("best_min_on_target").alias("min_on_target"),
        pl.col("best_max_off_target").alias("max_off_target"),
    )


def write_json_file(path: Path, value: Any) -> Path:
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
    resume_metadata: Mapping[str, Any] | None = None,
    config_filename: str = "run_config.json",
) -> list[Path]:
    """Write the original result contract plus sampled-population provenance."""

    written: list[Path] = []
    for table_name in ("evaluations", "final_population", "history"):
        table = insertion_only_table(getattr(optimizer, table_name))
        written.extend(write_table_pair(table, output_dir / table_name))

    compact_history = generation_min_on_max_off(optimizer.history)
    written.extend(write_table_pair(compact_history, output_dir / "generation_min_on_max_off"))

    written.append(write_json_file(output_dir / "summary.json", optimizer.summary()))
    written.append(
        write_json_file(output_dir / config_filename, config.as_dict())
    )

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

    best_row = optimizer.final_population.row(0, named=True)
    best_path = output_dir / "best_insertion_sequence.fasta"
    write_fasta(
        best_path,
        [
            (
                f"{best_row['candidate_id']}|rank=1|fitness={best_row['fitness']}",
                best_row["region_sequence"],
            )
        ],
    )
    written.append(best_path)

    if resume_metadata is None:
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
            write_json_file(
                output_dir / "initial_population_metadata.json",
                _sample_metadata(sampled_population),
            )
        )
    else:
        written.append(
            write_json_file(output_dir / "resume.json", resume_metadata)
        )
        if sampled_population:
            additions_path = output_dir / "resume_genome_additions.fasta"
            write_fasta(
                additions_path,
                (
                    (
                        f"addition_{index:04d}|source={sample.name or 'unnamed'}",
                        sample.sequence,
                    )
                    for index, sample in enumerate(sampled_population)
                ),
            )
            written.append(additions_path)
            addition_metadata = _sample_metadata(sampled_population)
            for item in addition_metadata:
                item["optimizer_role"] = "resume_genome_addition"
            written.append(
                write_json_file(
                    output_dir / "resume_genome_additions_metadata.json",
                    addition_metadata,
                )
            )

    region_length = optimizer.region.end - optimizer.region.start
    manifest_path = output_dir / "output_manifest.json"
    sequence_exports = {
        "region_length": region_length,
        "contains_materialized_plasmids": False,
        "evaluated_region_sequences.fasta": "all unique scored candidate regions",
        "final_population_region_sequences.fasta": "selected final candidate regions",
        "best_insertion_sequence.fasta": "rank-one candidate insertion",
    }
    if resume_metadata is None:
        sequence_exports["initial_population.fasta"] = (
            "genome-sampled starting promoters; sample zero is pasted into the plasmid"
        )
    elif sampled_population:
        sequence_exports["resume_genome_additions.fasta"] = (
            "genome-sampled insertions explicitly added to the restored population"
        )

    manifest = {
        "files": [path.name for path in written] + [manifest_path.name],
        "generation_min_on_max_off_semantics": {
            "population": "selected population after selection in each generation",
            "candidate": "best-fitness candidate using the optimizer's deterministic candidate-ID tie-break",
            "min_on_target": "minimum on-target score of the best-fitness candidate",
            "max_off_target": "maximum off-target score of the same best-fitness candidate",
            "same_candidate_required": True,
        },
        "sequence_exports": sequence_exports,
        "resume": dict(resume_metadata) if resume_metadata is not None else None,
    }
    written.append(write_json_file(manifest_path, manifest))
    return written

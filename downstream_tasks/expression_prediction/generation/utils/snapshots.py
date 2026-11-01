"""Generation population snapshots with insertion-only sequences."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

import polars as pl

from gena_expression import SequenceOptimizer

from .complexity import compute_frame_population_complexity
from .config import json_ready
from .results import write_fasta, write_table_pair


class PopulationSnapshotWriter:
    """Callable generation callback that writes configured population snapshots."""

    def __init__(
        self,
        output_dir: str | Path,
        generations: Iterable[int],
        *,
        skip_generations: Iterable[int] = (),
    ) -> None:
        self.output_dir = Path(output_dir) / "snapshots"
        self.generations = frozenset(generations)
        self.skip_generations = frozenset(skip_generations)
        self.written_paths: list[Path] = []

    @classmethod
    def from_config(
        cls,
        output_dir: str | Path,
        config: Mapping[str, Any],
        *,
        skip_generations: Iterable[int] = (),
    ) -> "PopulationSnapshotWriter":
        """Build a writer from the validated ``snapshots`` config section."""

        return cls(
            output_dir,
            config["generations"],
            skip_generations=skip_generations,
        )

    @staticmethod
    def _generation_metrics(
        generation: int,
        population: pl.DataFrame,
    ) -> dict[str, Any]:
        """Summarize fitness, champion scores, and insertion complexity."""

        fitness_values = [float(value) for value in population["fitness"]]
        best = population.row(0, named=True)
        best_scores = {
            column: best[column]
            for column in population.columns
            if column.startswith(("on_target__", "off_target__"))
        }
        on_scores = [
            float(value)
            for column, value in best_scores.items()
            if column.startswith("on_target__")
        ]
        off_scores = [
            float(value)
            for column, value in best_scores.items()
            if column.startswith("off_target__")
        ]
        return {
            "generation": generation,
            "optimization": {
                "population_size": population.height,
                "best_candidate_id": best["candidate_id"],
                "best_fitness": float(best["fitness"]),
                "mean_fitness": fmean(fitness_values),
                "min_fitness": min(fitness_values),
                "best_min_on_target": min(on_scores) if on_scores else None,
                "best_max_off_target": max(off_scores) if off_scores else None,
                "best_target_scores": best_scores,
            },
            "complexity": compute_frame_population_complexity(population),
        }

    def __call__(
        self,
        optimizer: SequenceOptimizer,
        generation: int,
    ) -> None:
        """Write one configured post-selection generation and otherwise do nothing."""

        if generation not in self.generations or generation in self.skip_generations:
            return

        population = optimizer.population_snapshot()
        if population.is_empty():
            raise RuntimeError(
                f"Cannot save generation {generation}: selected population is empty."
            )
        if "sequence" in population.columns:
            raise ValueError(
                "Population snapshots must not contain materialized plasmid sequences."
            )

        snapshot_dir = self.output_dir / f"generation_{generation:06d}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        written = write_table_pair(population, snapshot_dir / "population")
        written.append(
            write_fasta(
                snapshot_dir / "insertion_sequences.fasta",
                (
                    (
                        f"{row['candidate_id']}|rank={row['rank']}|fitness={row['fitness']}",
                        row["region_sequence"],
                    )
                    for row in population.iter_rows(named=True)
                ),
            )
        )

        metrics_path = snapshot_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                json_ready(self._generation_metrics(generation, population)),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        written.append(metrics_path)
        self.written_paths.extend(written)

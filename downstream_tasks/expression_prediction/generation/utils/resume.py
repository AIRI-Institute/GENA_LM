"""Load and validate insertion-only populations for warm restarts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl


class ResumeError(ValueError):
    """Raised when a saved population cannot be resumed safely."""


@dataclass(frozen=True)
class ResumePopulation:
    """One validated post-selection population loaded from a snapshot."""

    generation: int
    insertion_sequences: tuple[str, ...]
    snapshot_directory: Path
    configuration_verified: bool

    def metadata(self, *, genome_sequences_added: int) -> dict[str, Any]:
        """Return concise, JSON-ready resume provenance."""

        return {
            "resumed": True,
            "source_snapshot": str(self.snapshot_directory),
            "restored_generation": self.generation,
            "restored_population_size": len(self.insertion_sequences),
            "genome_sequences_added": genome_sequences_added,
            "scores_recomputed": True,
            "configuration_verified": self.configuration_verified,
        }


_GENERATION_DIRECTORY = re.compile(r"generation_(\d+)$")


def _latest_complete_snapshot(source_directory: Path) -> tuple[Path, int]:
    """Return the newest directory with the two canonical resume files."""

    snapshots = source_directory / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"Snapshot directory does not exist: {snapshots}")

    complete: list[tuple[int, Path]] = []
    for path in snapshots.iterdir():
        match = _GENERATION_DIRECTORY.fullmatch(path.name)
        if (
            path.is_dir()
            and match is not None
            and (path / "population.parquet").is_file()
            and (path / "metrics.json").is_file()
        ):
            complete.append((int(match.group(1)), path))
    if not complete:
        raise FileNotFoundError(
            f"No complete population snapshots were found under {snapshots}."
        )
    generation, path = max(complete)
    return path, generation


def _compatibility_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep settings that define scoring and evolutionary behavior."""

    model = dict(config["model"])
    model.pop("device", None)
    optimization = config["optimization"]
    return {
        "model": model,
        "plasmid": config["plasmid"],
        "conditions": config["conditions"],
        "scorer": config["scorer"],
        "fitness": config["fitness"],
        "offspring": config["offspring"],
        "optimization": {
            key: optimization[key]
            for key in (
                "population_size",
                "offspring_size",
                "selection_strategy",
                "elite_count",
                "tournament_size",
                "offspring_generator_mode",
                "random_seed",
            )
        },
    }


def _verify_configuration(
    source_directory: Path,
    current_config: Mapping[str, Any],
) -> bool:
    """Compare critical settings when the source run configuration exists."""

    path = next(
        (
            candidate
            for candidate in (
                source_directory / "run_config.json",
                source_directory / "resume_config.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if path is None:
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved_view = _compatibility_view(saved)
        current_view = _compatibility_view(current_config)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ResumeError(f"Cannot validate source run configuration: {path}") from exc
    if saved_view != current_view:
        changed = sorted(
            key for key in saved_view if saved_view[key] != current_view[key]
        )
        raise ResumeError(
            "Resume configuration differs from the source run in: "
            + ", ".join(changed)
        )
    return True


def load_latest_population(
    source_directory: str | Path,
    *,
    expected_sequence_length: int,
    expected_population_size: int,
    final_generation: int,
    allow_population_shortfall: bool,
    current_config: Mapping[str, Any],
) -> ResumePopulation:
    """Load the latest complete snapshot and validate its resume contract."""

    source = Path(source_directory)
    snapshot, generation = _latest_complete_snapshot(source)
    if generation >= final_generation:
        raise ResumeError(
            f"Snapshot generation {generation} must be below "
            f"optimization.n_generations ({final_generation})."
        )

    try:
        metrics = json.loads(
            (snapshot / "metrics.json").read_text(encoding="utf-8")
        )
        population = pl.read_parquet(snapshot / "population.parquet")
    except (json.JSONDecodeError, OSError, pl.exceptions.PolarsError) as exc:
        raise ResumeError(f"Cannot read complete snapshot: {snapshot}") from exc

    if not isinstance(metrics, Mapping):
        raise ResumeError("Snapshot metrics.json must contain an object.")
    if metrics.get("generation") != generation:
        raise ResumeError(
            f"Snapshot directory generation {generation} does not match metrics.json."
        )
    if population.is_empty():
        raise ResumeError("Saved population is empty.")
    if "sequence" in population.columns:
        raise ResumeError("Resume snapshots must not contain plasmid sequences.")
    if "region_sequence" not in population.columns:
        raise ResumeError("Saved population has no region_sequence column.")
    if population.height > expected_population_size or (
        population.height != expected_population_size
        and not allow_population_shortfall
    ):
        raise ResumeError(
            f"Saved population contains {population.height} sequences; expected "
            f"{expected_population_size}."
        )

    if "rank" in population.columns:
        population = population.sort("rank")
    sequences = tuple(population["region_sequence"].to_list())
    if any(not isinstance(sequence, str) for sequence in sequences):
        raise ResumeError("Every saved region_sequence must be a string.")
    if len(set(sequences)) != len(sequences):
        raise ResumeError("Saved population contains duplicate insertion sequences.")
    for sequence in sequences:
        if len(sequence) != expected_sequence_length:
            raise ResumeError(
                f"Saved insertion length {len(sequence)} does not match configured "
                f"length {expected_sequence_length}."
            )
        invalid = sorted(set(sequence) - set("ACGTN"))
        if invalid:
            raise ResumeError(
                "Saved insertion contains unsupported DNA symbols: "
                + ", ".join(invalid)
            )

    if "candidate_id" in population.columns:
        expected_ids = [
            hashlib.sha256(sequence.encode("ascii")).hexdigest()
            for sequence in sequences
        ]
        if population["candidate_id"].to_list() != expected_ids:
            raise ResumeError(
                "Saved candidate IDs do not match their insertion sequences."
            )

    metric_optimization = metrics.get("optimization", {})
    if not isinstance(metric_optimization, Mapping):
        raise ResumeError("Snapshot optimization metrics must contain an object.")
    metric_size = metric_optimization.get("population_size")
    if metric_size is not None and metric_size != population.height:
        raise ResumeError(
            "metrics.json population size does not match population.parquet."
        )

    return ResumePopulation(
        generation=generation,
        insertion_sequences=sequences,
        snapshot_directory=snapshot,
        configuration_verified=_verify_configuration(source, current_config),
    )

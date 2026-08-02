"""Evolutionary optimization over one fixed-length annotated-sequence region."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from itertools import islice
from numbers import Real
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from ..sequences import AnnotatedSequence, Feature


def _require_polars():
    """Import the optional table dependency with a useful error message."""

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "SequenceOptimizer requires Polars. Install gena-expression with the "
            "'tables' extra."
        ) from exc
    return pl


class SequenceOptimizer:
    """Evolve and score fixed-length replacements for one sequence region.

    The default offspring generator receives only the selected parent region
    strings. Set ``offspring_generator_mode="records"`` to instead receive
    parent dictionaries and a second context dictionary containing the current
    generation, requested offspring count, region information, and seeded RNG.

    Scores are absolute single-sequence scores. The scorer must implement
    ``score_prediction(prediction)`` and return one finite scalar score.
    """

    def __init__(
        self,
        sequence: AnnotatedSequence,
        *,
        region: tuple[int, int] | str | Feature,
        tokenization_center: int | str | Feature,
        scorer: Any,
        on_targets: Mapping[str, Any] | Sequence[Any],
        off_targets: Mapping[str, Any] | Sequence[Any] = (),
        offspring_generator: Callable[..., Iterable[Any]],
        fitness_function: (
            Callable[[Mapping[str, float], Mapping[str, float]], float] | None
        ) = None,
        population_size: int = 64,
        offspring_size: int | None = None,
        n_generations: int = 20,
        initial_population: Iterable[str] | None = None,
        selection_strategy: Literal[
            "top_k",
            "tournament",
            "rank_weighted",
        ] = "top_k",
        elite_count: int = 1,
        tournament_size: int = 3,
        offspring_generator_mode: Literal["sequences", "records"] = "sequences",
        random_seed: int | None = 0,
    ) -> None:
        """Configure an evolutionary sequence-optimization run.

        ``initial_population`` and every generated string represent only the
        evolvable region. Their lengths must equal the original region length.
        Generation zero always includes the original region.
        """

        if not isinstance(sequence, AnnotatedSequence):
            raise TypeError("sequence must be an AnnotatedSequence.")
        if not sequence.sequence:
            raise ValueError("sequence must not be empty.")
        if not callable(getattr(scorer, "score_prediction", None)):
            raise TypeError("scorer must implement score_prediction(prediction).")
        if not callable(offspring_generator):
            raise TypeError("offspring_generator must be callable.")

        self._validate_positive_int(population_size, name="population_size")
        if offspring_size is not None:
            self._validate_positive_int(offspring_size, name="offspring_size")
        if isinstance(n_generations, bool) or not isinstance(n_generations, int):
            raise TypeError("n_generations must be an integer.")
        if n_generations < 0:
            raise ValueError("n_generations must be non-negative.")
        if isinstance(elite_count, bool) or not isinstance(elite_count, int):
            raise TypeError("elite_count must be an integer.")
        if not 0 <= elite_count <= population_size:
            raise ValueError("elite_count must be between 0 and population_size.")
        self._validate_positive_int(tournament_size, name="tournament_size")
        if selection_strategy not in {"top_k", "tournament", "rank_weighted"}:
            raise ValueError(
                "selection_strategy must be 'top_k', 'tournament', or "
                "'rank_weighted'."
            )
        if offspring_generator_mode not in {"sequences", "records"}:
            raise ValueError(
                "offspring_generator_mode must be 'sequences' or 'records'."
            )

        self.sequence = sequence
        self.region = self._resolve_region(sequence, region)
        self.tokenization_center = tokenization_center
        self.scorer = scorer
        self.on_targets = self._normalize_targets(on_targets, role="on_target")
        self.off_targets = self._normalize_targets(
            off_targets,
            role="off_target",
        )
        if not self.on_targets and not self.off_targets:
            raise ValueError("At least one on-target or off-target is required.")

        self.offspring_generator = offspring_generator
        self.fitness_function = fitness_function or self._default_fitness
        if not callable(self.fitness_function):
            raise TypeError("fitness_function must be callable.")
        self.population_size = population_size
        self.offspring_size = offspring_size or population_size
        self.n_generations = n_generations
        self.initial_population = tuple(initial_population or ())
        self.selection_strategy = selection_strategy
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.offspring_generator_mode = offspring_generator_mode
        self.random_seed = random_seed

        self._records: dict[str, dict[str, Any]] = {}
        self._population: list[str] = []
        self._history_rows: list[dict[str, Any]] = []
        self._rng = random.Random(random_seed)
        self._has_run = False

        pl = _require_polars()
        self.evaluations = pl.DataFrame()
        self.final_population = pl.DataFrame()
        self.history = pl.DataFrame()

    @staticmethod
    def _validate_positive_int(value: int, *, name: str) -> None:
        """Validate one strictly positive integer argument."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    @classmethod
    def _resolve_region(
        cls,
        sequence: AnnotatedSequence,
        region: tuple[int, int] | str | Feature,
    ) -> Feature:
        """Resolve a coordinate tuple, feature, or unique feature name."""

        if isinstance(region, str):
            selector = region.split(":", 1)[1] if region.startswith("feature:") else region
            matches = [
                feature
                for feature in sequence.features
                if feature.name == selector or feature.type == selector
            ]
            if not matches:
                raise KeyError(
                    f"Feature {selector!r} was not found on sequence "
                    f"{sequence.name!r}."
                )
            if len(matches) > 1:
                summary = ", ".join(
                    f"{feature.name}({feature.start}:{feature.end}, "
                    f"type={feature.type})"
                    for feature in matches
                )
                raise ValueError(
                    f"Feature selector {selector!r} matched multiple features: "
                    f"{summary}"
                )
            resolved = matches[0]
        elif isinstance(region, Feature):
            resolved = region
        elif (
            isinstance(region, tuple)
            and len(region) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in region
            )
        ):
            resolved = Feature(
                "optimization_region",
                region[0],
                region[1],
                type="optimization_region",
                source="optimization",
            )
        else:
            raise TypeError(
                "region must be a (start, end) tuple, Feature, or feature name."
            )

        if (
            resolved.start < 0
            or resolved.end > len(sequence)
            or resolved.end <= resolved.start
        ):
            raise ValueError(
                f"Optimization region {resolved.start}:{resolved.end} must be "
                f"non-empty and within sequence length {len(sequence)}."
            )
        return resolved

    @staticmethod
    def _normalize_targets(
        targets: Mapping[str, Any] | Sequence[Any],
        *,
        role: str,
    ) -> tuple[dict[str, Any], ...]:
        """Normalize target conditions to unique labeled records."""

        if isinstance(targets, Mapping):
            items = list(targets.items())
        else:
            if isinstance(targets, (str, bytes)):
                raise TypeError(f"{role}s must be a mapping or sequence of conditions.")
            items = []
            for index, condition in enumerate(targets):
                label = str(getattr(condition, "name", "") or f"{role}_{index}")
                items.append((label, condition))

        records: list[dict[str, Any]] = []
        labels: set[str] = set()
        for label, condition in items:
            normalized_label = str(label).strip()
            if not normalized_label:
                raise ValueError(f"{role} labels must not be empty.")
            if normalized_label in labels:
                raise ValueError(
                    f"{role} label {normalized_label!r} is not unique."
                )
            labels.add(normalized_label)
            records.append(
                {
                    "label": normalized_label,
                    "condition": condition,
                    "column": f"{role}__{normalized_label}",
                }
            )
        return tuple(records)

    @staticmethod
    def _default_fitness(
        on_scores: Mapping[str, float],
        off_scores: Mapping[str, float],
    ) -> float:
        """Return mean on-target score minus mean off-target score."""

        on_value = mean(on_scores.values()) if on_scores else 0.0
        off_value = mean(off_scores.values()) if off_scores else 0.0
        return float(on_value - off_value)

    def _validate_region_sequence(self, region_sequence: Any) -> str:
        """Normalize and validate one generated fixed-length DNA region."""

        if not isinstance(region_sequence, str):
            raise TypeError("Generated region sequences must be strings.")
        normalized = region_sequence.upper()
        expected_length = self.region.end - self.region.start
        if len(normalized) != expected_length:
            raise ValueError(
                f"Generated region length {len(normalized)} does not match "
                f"required length {expected_length}."
            )
        invalid = sorted(set(normalized) - set("ACGTN"))
        if invalid:
            raise ValueError(
                "Generated region contains unsupported DNA symbols: "
                + ", ".join(invalid)
            )
        return normalized

    @staticmethod
    def _candidate_id(region_sequence: str) -> str:
        """Return a stable identifier for one region sequence."""

        return hashlib.sha256(region_sequence.encode("ascii")).hexdigest()

    def _new_record(
        self,
        region_sequence: str,
        *,
        generation: int,
        source: str,
        parent_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Create one unscored internal candidate record."""

        return {
            "candidate_id": self._candidate_id(region_sequence),
            "region_sequence": region_sequence,
            "generation_born": generation,
            "source": source,
            "parent_ids": list(parent_ids),
            "on_scores": {},
            "off_scores": {},
            "fitness": None,
        }

    def _materialize_region(
        self,
        region_sequence: str,
        *,
        candidate_id: str,
    ) -> AnnotatedSequence:
        """Insert one region while preserving annotations and coordinates."""

        full_sequence = (
            self.sequence.sequence[: self.region.start]
            + region_sequence
            + self.sequence.sequence[self.region.end :]
        )
        name = self.sequence.name or "sequence"
        return replace(
            self.sequence,
            sequence=full_sequence,
            name=f"{name}:{candidate_id[:12]}",
            metadata={
                **dict(self.sequence.metadata),
                "optimization_candidate_id": candidate_id,
                "optimization_region": self.region.name,
            },
        )

    def materialize(self, candidate_id: str) -> AnnotatedSequence:
        """Return one evaluated candidate as an :class:`AnnotatedSequence`."""

        try:
            record = self._records[candidate_id]
        except KeyError as exc:
            raise KeyError(f"Unknown optimization candidate {candidate_id!r}.") from exc
        return self._materialize_region(
            record["region_sequence"],
            candidate_id=candidate_id,
        )

    @property
    def best_sequence(self) -> AnnotatedSequence:
        """Return the highest-fitness sequence in the final population."""

        if not self._population:
            raise RuntimeError("No final population is available; run the optimizer first.")
        best_id = min(self._population, key=self._fitness_sort_key)
        return self.materialize(best_id)

    def _score_candidates(
        self,
        model: Any,
        candidate_ids: Sequence[str],
        *,
        evaluation_chunk_size: int,
        grouping: Literal[
            "serial",
            "no_grouping",
            "condition",
            "sequence",
            "auto",
        ],
        preprocessing_workers: int,
        preprocessing_backend: Literal["process", "thread"],
        max_records_per_forward: int | None,
        prefetch_batches: int,
        show_progress: bool,
    ) -> None:
        """Predict and score each candidate under every configured target."""

        pending = [
            candidate_id
            for candidate_id in candidate_ids
            if self._records[candidate_id]["fitness"] is None
        ]
        if not pending:
            return

        starts: Any = range(0, len(pending), evaluation_chunk_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                starts = tqdm(
                    starts,
                    total=math.ceil(len(pending) / evaluation_chunk_size),
                    desc="Optimizing sequences",
                    unit="chunk",
                )
            except ImportError:
                pass

        targets = [
            ("on", target)
            for target in self.on_targets
        ] + [
            ("off", target)
            for target in self.off_targets
        ]
        requires_tokens = bool(getattr(self.scorer, "requires_tokens", True))

        for chunk_start in starts:
            chunk_ids = pending[
                chunk_start : chunk_start + evaluation_chunk_size
            ]
            materialized = {
                candidate_id: self.materialize(candidate_id)
                for candidate_id in chunk_ids
            }
            sequences = []
            conditions = []
            score_keys = []
            for candidate_id in chunk_ids:
                candidate = materialized[candidate_id]
                for role, target in targets:
                    # Reuse the same object across conditions so tokenization
                    # can be deduplicated by the existing model path.
                    sequences.append(candidate)
                    conditions.append(target["condition"])
                    score_keys.append((candidate_id, role, target["label"]))

            predictions = model.predict_multiple_sequences(
                sequences,
                conditions=conditions,
                center=self.tokenization_center,
                grouping=grouping,
                preprocessing_workers=preprocessing_workers,
                preprocessing_backend=preprocessing_backend,
                max_records_per_forward=max_records_per_forward,
                prefetch_batches=prefetch_batches,
                show_progress=False,
                return_tokens=requires_tokens,
                retention="full" if requires_tokens else "scalars",
            )
            if len(predictions) != len(score_keys):
                raise RuntimeError(
                    f"Model returned {len(predictions)} predictions for "
                    f"{len(score_keys)} sequence-condition rows."
                )

            for prediction, (candidate_id, role, label) in zip(
                predictions,
                score_keys,
            ):
                result = self.scorer.score_prediction(prediction)
                score = getattr(result, "score", None)
                if isinstance(score, bool) or not isinstance(score, Real):
                    raise TypeError(
                        "SequenceOptimizer requires one scalar score per "
                        f"condition; scorer returned {type(score).__name__}."
                    )
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError(
                        f"Scorer returned a non-finite value for candidate "
                        f"{candidate_id!r}, condition {label!r}."
                    )
                score_map = (
                    self._records[candidate_id]["on_scores"]
                    if role == "on"
                    else self._records[candidate_id]["off_scores"]
                )
                score_map[label] = score

            for candidate_id in chunk_ids:
                record = self._records[candidate_id]
                fitness = self.fitness_function(
                    dict(record["on_scores"]),
                    dict(record["off_scores"]),
                )
                if isinstance(fitness, bool) or not isinstance(fitness, Real):
                    raise TypeError("fitness_function must return one real number.")
                fitness = float(fitness)
                if not math.isfinite(fitness):
                    raise ValueError("fitness_function must return a finite value.")
                record["fitness"] = fitness

    def _fitness_sort_key(self, candidate_id: str) -> tuple[float, str]:
        """Return a deterministic descending-fitness sort key."""

        fitness = self._records[candidate_id]["fitness"]
        if fitness is None:
            raise RuntimeError(f"Candidate {candidate_id!r} has not been scored.")
        return -float(fitness), candidate_id

    def _select_population(self, candidate_ids: Sequence[str]) -> list[str]:
        """Select the next population with optional protected elites."""

        ordered = sorted(set(candidate_ids), key=self._fitness_sort_key)
        target_size = min(self.population_size, len(ordered))
        elite_size = min(self.elite_count, target_size)
        selected = ordered[:elite_size]
        available = ordered[elite_size:]

        if self.selection_strategy == "top_k":
            selected.extend(available[: target_size - len(selected)])
            return selected

        while len(selected) < target_size and available:
            if self.selection_strategy == "tournament":
                competitors = self._rng.sample(
                    available,
                    k=min(self.tournament_size, len(available)),
                )
                winner = min(competitors, key=self._fitness_sort_key)
            else:
                ranked = sorted(available, key=self._fitness_sort_key)
                weights = list(range(len(ranked), 0, -1))
                winner = self._rng.choices(ranked, weights=weights, k=1)[0]
            selected.append(winner)
            available.remove(winner)
        return selected

    def _parent_records(self) -> list[dict[str, Any]]:
        """Return compact parent dictionaries for an advanced generator."""

        return [
            {
                "candidate_id": candidate_id,
                "region_sequence": self._records[candidate_id]["region_sequence"],
                "fitness": self._records[candidate_id]["fitness"],
                "on_scores": dict(self._records[candidate_id]["on_scores"]),
                "off_scores": dict(self._records[candidate_id]["off_scores"]),
                "generation_born": self._records[candidate_id]["generation_born"],
                "parent_ids": list(self._records[candidate_id]["parent_ids"]),
            }
            for candidate_id in self._population
        ]

    def _generate_offspring(self, generation: int) -> tuple[list[str], list[str]]:
        """Generate proposed IDs and create records for previously unseen DNA."""

        if self.offspring_generator_mode == "sequences":
            parents = [
                self._records[candidate_id]["region_sequence"]
                for candidate_id in self._population
            ]
            generated = self.offspring_generator(parents)
        else:
            context = {
                "generation": generation,
                "n_generations": self.n_generations,
                "n_offspring": self.offspring_size,
                "population_size": self.population_size,
                "region": {
                    "name": self.region.name,
                    "start": self.region.start,
                    "end": self.region.end,
                    "length": self.region.end - self.region.start,
                },
                "rng": self._rng,
            }
            generated = self.offspring_generator(
                self._parent_records(),
                context,
            )

        proposed_ids: list[str] = []
        new_ids: list[str] = []
        single_parent_ids = self._population if len(self._population) == 1 else []
        for item in islice(generated, self.offspring_size):
            if isinstance(item, str):
                region_sequence = item
                parent_ids = single_parent_ids
            elif isinstance(item, Mapping):
                if "region_sequence" not in item:
                    raise KeyError(
                        "Generated mappings must contain 'region_sequence'."
                    )
                region_sequence = item["region_sequence"]
                parent_ids = list(item.get("parent_ids", ()))
            else:
                raise TypeError(
                    "offspring_generator must yield strings or mappings."
                )

            normalized = self._validate_region_sequence(region_sequence)
            candidate_id = self._candidate_id(normalized)
            if candidate_id not in proposed_ids:
                proposed_ids.append(candidate_id)
            if candidate_id in self._records:
                continue
            self._records[candidate_id] = self._new_record(
                normalized,
                generation=generation,
                source="offspring",
                parent_ids=parent_ids,
            )
            new_ids.append(candidate_id)
        return proposed_ids, new_ids

    def _add_history(
        self,
        *,
        generation: int,
        proposed: int,
        evaluated: int,
    ) -> None:
        """Record one compact generation summary."""

        fitness_values = [
            float(self._records[candidate_id]["fitness"])
            for candidate_id in self._population
        ]
        self._history_rows.append(
            {
                "generation": generation,
                "proposed_candidates": proposed,
                "evaluated_candidates": evaluated,
                "population_size": len(self._population),
                "best_fitness": max(fitness_values),
                "mean_fitness": mean(fitness_values),
            }
        )

    def _evaluation_row(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Flatten one internal candidate into a Polars-friendly row."""

        row = {
            "candidate_id": record["candidate_id"],
            "region_sequence": record["region_sequence"],
            "generation_born": record["generation_born"],
            "source": record["source"],
            "parent_ids": record["parent_ids"],
            "fitness": record["fitness"],
        }
        for target in self.on_targets:
            row[target["column"]] = record["on_scores"][target["label"]]
        for target in self.off_targets:
            row[target["column"]] = record["off_scores"][target["label"]]
        return row

    def _refresh_tables(self) -> None:
        """Rebuild public Polars tables from compact internal records."""

        pl = _require_polars()
        evaluation_rows = [
            self._evaluation_row(record)
            for record in self._records.values()
            if record["fitness"] is not None
        ]
        self.evaluations = (
            pl.DataFrame(evaluation_rows, strict=False).sort(
                ["generation_born", "candidate_id"]
            )
            if evaluation_rows
            else pl.DataFrame()
        )

        ordered_population = sorted(
            self._population,
            key=self._fitness_sort_key,
        )
        final_rows = []
        for rank, candidate_id in enumerate(ordered_population, start=1):
            record = self._records[candidate_id]
            row = self._evaluation_row(record)
            row["rank"] = rank
            row["sequence"] = self.materialize(candidate_id).sequence
            final_rows.append(row)
        self.final_population = (
            pl.DataFrame(final_rows, strict=False).select(
                "candidate_id",
                "sequence",
                "region_sequence",
                "rank",
                "generation_born",
                "source",
                "parent_ids",
                "fitness",
                *[
                    target["column"]
                    for target in (*self.on_targets, *self.off_targets)
                ],
            )
            if final_rows
            else pl.DataFrame()
        )
        self.history = (
            pl.DataFrame(self._history_rows, strict=False)
            if self._history_rows
            else pl.DataFrame()
        )

    def run_evolution(
        self,
        model: Any,
        *,
        evaluation_chunk_size: int = 128,
        grouping: Literal[
            "serial",
            "no_grouping",
            "condition",
            "sequence",
            "auto",
        ] = "auto",
        preprocessing_workers: int = 0,
        preprocessing_backend: Literal["process", "thread"] = "process",
        max_records_per_forward: int | None = None,
        prefetch_batches: int = 1,
        show_progress: bool = True,
    ) -> "SequenceOptimizer":
        """Run generation zero followed by ``n_generations`` reproduction rounds."""

        if self._has_run:
            raise RuntimeError("This optimizer has already run; call reset() first.")
        if not hasattr(model, "predict_multiple_sequences"):
            raise TypeError("model must provide predict_multiple_sequences(...).")
        self._validate_positive_int(
            evaluation_chunk_size,
            name="evaluation_chunk_size",
        )

        original_region = self.sequence.sequence[
            self.region.start : self.region.end
        ]
        seed_regions = [original_region, *self.initial_population]
        seed_ids: list[str] = []
        for index, region_sequence in enumerate(seed_regions):
            normalized = self._validate_region_sequence(region_sequence)
            candidate_id = self._candidate_id(normalized)
            if candidate_id in self._records:
                continue
            self._records[candidate_id] = self._new_record(
                normalized,
                generation=0,
                source="reference" if index == 0 else "initial",
            )
            seed_ids.append(candidate_id)

        score_options = {
            "evaluation_chunk_size": evaluation_chunk_size,
            "grouping": grouping,
            "preprocessing_workers": preprocessing_workers,
            "preprocessing_backend": preprocessing_backend,
            "max_records_per_forward": max_records_per_forward,
            "prefetch_batches": prefetch_batches,
            "show_progress": show_progress,
        }
        self._score_candidates(model, seed_ids, **score_options)
        self._population = self._select_population(seed_ids)
        self._add_history(
            generation=0,
            proposed=len(seed_ids),
            evaluated=len(seed_ids),
        )

        for generation in range(1, self.n_generations + 1):
            proposed_ids, new_ids = self._generate_offspring(generation)
            self._score_candidates(model, new_ids, **score_options)
            pool = [*self._population, *proposed_ids]
            self._population = self._select_population(pool)
            self._add_history(
                generation=generation,
                proposed=len(proposed_ids),
                evaluated=len(new_ids),
            )

        self._has_run = True
        self._refresh_tables()
        return self

    def top(
        self,
        n: int = 20,
        *,
        table: Literal["final_population", "evaluations"] = "final_population",
    ):
        """Return the highest-fitness rows from a public result table."""

        self._validate_positive_int(n, name="n")
        if table == "final_population":
            frame = self.final_population
        elif table == "evaluations":
            frame = self.evaluations
        else:
            raise ValueError(
                "table must be 'final_population' or 'evaluations'."
            )
        if frame.is_empty():
            return frame
        return frame.sort(
            ["fitness", "candidate_id"],
            descending=[True, False],
        ).head(n)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the current optimization state."""

        best_fitness = None
        if self._population:
            best_id = min(self._population, key=self._fitness_sort_key)
            best_fitness = self._records[best_id]["fitness"]
        return {
            "sequence_name": self.sequence.name,
            "region": {
                "name": self.region.name,
                "start": self.region.start,
                "end": self.region.end,
            },
            "selection_strategy": self.selection_strategy,
            "generations": self.n_generations,
            "evaluated_candidates": len(self._records),
            "final_population": len(self._population),
            "best_fitness": best_fitness,
        }

    def write_parquet(
        self,
        path: str | Path,
        *,
        table: Literal[
            "final_population",
            "evaluations",
            "history",
        ] = "final_population",
    ) -> Path:
        """Write one public result table to a Parquet file."""

        frame = getattr(self, table)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output)
        return output

    def reset(self) -> "SequenceOptimizer":
        """Clear results and restore the configured random seed."""

        pl = _require_polars()
        self._records.clear()
        self._population.clear()
        self._history_rows.clear()
        self._rng = random.Random(self.random_seed)
        self._has_run = False
        self.evaluations = pl.DataFrame()
        self.final_population = pl.DataFrame()
        self.history = pl.DataFrame()
        return self

"""In-silico mutagenesis over annotated sequence regions."""

from __future__ import annotations

import math
import random
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from ..sequences import AnnotatedSequence, Feature, SequencePair
from ..variants import Variant


_CENTER_FEATURE = "__ism_center__"
_STRUCTURAL_COLUMNS = (
    "variant_id",
    "mutation_type",
    "position",
    "end",
    "position_in_region",
    "ref",
    "alt",
    "length",
    "source",
    "source_position",
    "source_chrom",
    "source_strand",
)
_DNA_ALPHABET = "ACGT"


def _require_polars():
    """Import the optional table dependency with a useful error message."""

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "ISM requires Polars. Install gena-expression with the 'tables' extra."
        ) from exc
    return pl


class ISM:
    """Generate, retain, score, and plot an in-silico mutation library.

    Coordinates are 0-based and half-open, relative to ``sequence``. The
    mutation table stores compact allele descriptions; full alternative
    sequences are materialized lazily while scoring.
    """

    def __init__(
        self,
        sequence: AnnotatedSequence,
        *,
        center: int | str | Feature,
        region: tuple[int, int] | str | Feature,
        condition: Any,
        n_substitutions: int | None = None,
        n_deletions: int | None = None,
        random_seed: int | None = 0,
        center_overlap: Literal["nearest", "error"] = "nearest",
    ) -> None:
        """Create substitutions and optional single-base deletions.

        ``n_substitutions=None`` creates all three SNVs at every A/C/G/T base.
        A finite count is balanced across positions before a position receives
        its second or third alternative. ``n_deletions=None`` disables
        deletions; otherwise that many distinct single-base deletions are
        sampled uniformly across the region.
        """

        if not isinstance(sequence, AnnotatedSequence):
            raise TypeError("sequence must be an AnnotatedSequence.")
        if not sequence.sequence:
            raise ValueError("sequence must not be empty.")
        if center_overlap not in {"nearest", "error"}:
            raise ValueError("center_overlap must be 'nearest' or 'error'.")
        if any(feature.name == _CENTER_FEATURE for feature in sequence.features):
            raise ValueError(f"Sequence already contains reserved feature {_CENTER_FEATURE!r}.")
        if any(feature.name == "variant" or feature.type == "variant" for feature in sequence.features):
            raise ValueError(
                "ISM requires a reference sequence without an existing 'variant' feature."
            )

        self.sequence = sequence
        self.region = self._resolve_region(sequence, region)
        self.center = center
        self.center_position = self._resolve_center(sequence, center)
        self.condition = condition
        self.random_seed = random_seed
        self.center_overlap = center_overlap
        self.n_substitutions = self._validate_count(
            n_substitutions,
            name="n_substitutions",
            allow_none=True,
        )
        self.n_deletions = self._validate_count(
            n_deletions,
            name="n_deletions",
            allow_none=True,
        )

        records = self._build_records()
        pl = _require_polars()
        schema = {
            "variant_id": pl.Utf8,
            "mutation_type": pl.Utf8,
            "position": pl.Int64,
            "end": pl.Int64,
            "position_in_region": pl.Int64,
            "ref": pl.Utf8,
            "alt": pl.Utf8,
            "length": pl.Int64,
            "source": pl.Utf8,
            "source_position": pl.Int64,
            "source_chrom": pl.Utf8,
            "source_strand": pl.Utf8,
        }
        self.variants = pl.DataFrame(records, schema=schema)

    @property
    def scorer_names(self) -> tuple[str, ...]:
        """Return score columns currently attached to the mutation table."""

        return tuple(
            column
            for column in self.variants.columns
            if column not in _STRUCTURAL_COLUMNS and not column.endswith("__error")
        )

    @staticmethod
    def _validate_count(value: int | None, *, name: str, allow_none: bool) -> int | None:
        if value is None and allow_none:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer or None.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
        return value

    @classmethod
    def _resolve_feature(cls, sequence: AnnotatedSequence, selector: str) -> Feature:
        if selector.startswith("feature:"):
            selector = selector.split(":", 1)[1]
        matches = [
            feature
            for feature in sequence.features
            if feature.name == selector or feature.type == selector
        ]
        if not matches:
            raise KeyError(f"Feature {selector!r} was not found on sequence {sequence.name!r}.")
        if len(matches) > 1:
            summary = ", ".join(
                f"{feature.name}({feature.start}:{feature.end}, type={feature.type})"
                for feature in matches
            )
            raise ValueError(f"Feature selector {selector!r} matched multiple features: {summary}")
        return matches[0]

    @classmethod
    def _resolve_region(
        cls,
        sequence: AnnotatedSequence,
        region: tuple[int, int] | str | Feature,
    ) -> Feature:
        if isinstance(region, str):
            resolved = cls._resolve_feature(sequence, region)
        elif isinstance(region, Feature):
            resolved = region
        elif (
            isinstance(region, tuple)
            and len(region) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in region)
        ):
            resolved = Feature("ism_region", region[0], region[1], type="ism_region", source="ism")
        else:
            raise TypeError("region must be a (start, end) tuple, Feature, or feature name.")
        if resolved.start < 0 or resolved.end > len(sequence) or resolved.end <= resolved.start:
            raise ValueError(
                f"ISM region {resolved.start}:{resolved.end} must be non-empty and within "
                f"sequence length {len(sequence)}."
            )
        return resolved

    @classmethod
    def _resolve_center(cls, sequence: AnnotatedSequence, center: int | str | Feature) -> int:
        if isinstance(center, bool):
            raise TypeError("center must be an integer, Feature, or feature name.")
        if isinstance(center, int):
            position = center
        elif isinstance(center, str):
            feature = cls._resolve_feature(sequence, center)
            position = (feature.start + feature.end) // 2
        elif isinstance(center, Feature):
            position = (center.start + center.end) // 2
        else:
            raise TypeError("center must be an integer, Feature, or feature name.")
        if not 0 <= position < len(sequence):
            raise ValueError(
                f"Resolved center {position} must be within sequence length {len(sequence)}."
            )
        return position

    def _build_records(self) -> list[dict[str, Any]]:
        substitutions = self._substitution_records()
        deletions = self._deletion_records()
        records = substitutions + deletions
        records.sort(key=lambda row: (row["position"], row["mutation_type"], row["alt"]))
        return records

    def _substitution_records(self) -> list[dict[str, Any]]:
        positions = [
            position
            for position in range(self.region.start, self.region.end)
            if self.sequence.sequence[position] in _DNA_ALPHABET
        ]
        total = 3 * len(positions)
        if self.n_substitutions is not None and self.n_substitutions > total:
            raise ValueError(
                f"n_substitutions={self.n_substitutions} exceeds the {total} available SNVs."
            )

        if self.n_substitutions is None:
            selected = [
                (position, alt)
                for position in positions
                for alt in _DNA_ALPHABET
                if alt != self.sequence.sequence[position]
            ]
        else:
            rng = random.Random(self.random_seed)
            alternatives: dict[int, list[str]] = {}
            for position in positions:
                choices = [base for base in _DNA_ALPHABET if base != self.sequence.sequence[position]]
                rng.shuffle(choices)
                alternatives[position] = choices
            selected = []
            for alternative_index in range(3):
                round_positions = list(positions)
                rng.shuffle(round_positions)
                for position in round_positions:
                    if len(selected) == self.n_substitutions:
                        break
                    selected.append((position, alternatives[position][alternative_index]))
                if len(selected) == self.n_substitutions:
                    break

        return [
            self._record(
                variant_id=f"sub_{position}_{self.sequence.sequence[position]}_{alt}",
                mutation_type="substitution",
                position=position,
                ref=self.sequence.sequence[position],
                alt=alt,
            )
            for position, alt in selected
        ]

    def _deletion_records(self) -> list[dict[str, Any]]:
        if self.n_deletions is None or self.n_deletions == 0:
            return []
        positions = list(range(self.region.start, self.region.end))
        if self.n_deletions > len(positions):
            raise ValueError(
                f"n_deletions={self.n_deletions} exceeds the {len(positions)} available "
                "single-base deletions."
            )
        deletion_seed = None if self.random_seed is None else self.random_seed + 1
        selected = random.Random(deletion_seed).sample(positions, self.n_deletions)
        return [
            self._record(
                variant_id=f"del_{position}_{self.sequence.sequence[position]}",
                mutation_type="deletion",
                position=position,
                ref=self.sequence.sequence[position],
                alt="",
            )
            for position in selected
        ]

    def _record(
        self,
        *,
        variant_id: str,
        mutation_type: str,
        position: int,
        ref: str,
        alt: str,
    ) -> dict[str, Any]:
        source_coordinate = self.sequence.coordinate_map.seq_to_source(position)
        return {
            "variant_id": variant_id,
            "mutation_type": mutation_type,
            "position": position,
            "end": position + 1,
            "position_in_region": position - self.region.start,
            "ref": ref,
            "alt": alt,
            "length": 1,
            "source": source_coordinate.source if source_coordinate else None,
            "source_position": source_coordinate.position if source_coordinate else None,
            "source_chrom": source_coordinate.chrom if source_coordinate else None,
            "source_strand": source_coordinate.strand if source_coordinate else None,
        }

    def _row(self, variant_id: str) -> dict[str, Any]:
        pl = _require_polars()
        rows = self.variants.filter(pl.col("variant_id") == variant_id)
        if rows.height == 0:
            raise KeyError(f"Unknown ISM variant {variant_id!r}.")
        if rows.height > 1:
            raise RuntimeError(f"Variant identifier {variant_id!r} is not unique.")
        return rows.row(0, named=True)

    def variant(self, variant_id: str) -> Variant:
        """Return one compact table row as a :class:`Variant`."""

        return self._variant_from_row(self._row(variant_id))

    def _variant_from_row(self, row: Mapping[str, Any]) -> Variant:
        return Variant(
            chrom=None,
            pos=int(row["position"]),
            ref=str(row["ref"]),
            alt=str(row["alt"]),
            id=str(row["variant_id"]),
            metadata={
                "mutation_type": str(row["mutation_type"]),
                "position_in_region": int(row["position_in_region"]),
                "region_name": self.region.name,
                "source": row.get("source"),
                "source_position": row.get("source_position"),
                "source_chrom": row.get("source_chrom"),
                "source_strand": row.get("source_strand"),
            },
        )

    def sequence_pair(self, variant_id: str) -> SequencePair:
        """Materialize one annotated reference/alternative sequence pair."""

        return self._sequence_pair_from_row(self._row(variant_id))

    def _sequence_pair_from_row(self, row: Mapping[str, Any]) -> SequencePair:
        variant = self._variant_from_row(row)
        position = int(row["position"])
        ref = self.sequence.add_feature(
            "variant",
            position,
            position + 1,
            type="variant",
            source="variant",
            metadata=variant.to_dict(),
        )
        alt = variant.apply_to(self.sequence, offset=position)

        # Variant.apply_to marks a deleted final base outside the shorter allele.
        # Replace that generated marker with a valid one-base boundary marker.
        alt = alt.remove_features(name="variant", source="variant")
        alt_marker = min(position, len(alt) - 1)
        alt = alt.add_feature(
            "variant",
            alt_marker,
            alt_marker + 1,
            type="variant",
            source="variant",
            metadata=variant.to_dict(),
        )

        alt_center = self.center_position
        if row["mutation_type"] == "deletion":
            if position < self.center_position:
                alt_center -= 1
            elif position == self.center_position:
                if self.center_overlap == "error":
                    raise ValueError(
                        f"Deletion {variant.id!r} removes the tokenization center at "
                        f"position {self.center_position}."
                    )
                alt_center = min(position, len(alt) - 1)

        ref = ref.add_feature(
            _CENTER_FEATURE,
            self.center_position,
            self.center_position + 1,
            type="tokenization_center",
            source="ism",
        )
        alt = alt.add_feature(
            _CENTER_FEATURE,
            alt_center,
            alt_center + 1,
            type="tokenization_center",
            source="ism",
        )
        pair = SequencePair(
            ref=ref,
            alt=alt,
            variant=variant,
            metadata={"name": variant.id, "ism_region": self.region.name},
        )
        pair.assert_compatible()
        return pair

    @staticmethod
    def _scorer_columns(scorer: Any, column_name: str | None) -> tuple[str, ...]:
        scorers = getattr(scorer, "scorers", None)
        if scorers is not None:
            if column_name is not None:
                raise ValueError("column_name cannot be used with a ScorerSet.")
            if isinstance(scorers, Mapping):
                names = tuple(str(name) for name in scorers)
            else:
                names = tuple(str(getattr(item, "name", "")) for item in scorers)
        else:
            names = (str(column_name or getattr(scorer, "name", type(scorer).__name__)),)
        if not names or any(not name for name in names):
            raise ValueError("Every scorer must have a non-empty name.")
        if len(set(names)) != len(names):
            raise ValueError(f"Scorer names must be unique; received {names!r}.")
        for name in names:
            if name in _STRUCTURAL_COLUMNS or name.endswith("__error"):
                raise ValueError(f"Scorer name {name!r} conflicts with an ISM table column.")
        return names

    @staticmethod
    def _scalar_score(value: Any, *, scorer_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"ISM dataframe columns require scalar scores; scorer {scorer_name!r} "
                f"returned {type(value).__name__}."
            )
        return float(value)

    @classmethod
    def _scores_from_result(
        cls,
        result: Any,
        scorer_columns: Sequence[str],
    ) -> dict[str, float]:
        report_results = getattr(result, "results", None)
        if report_results is None:
            if len(scorer_columns) != 1 or not hasattr(result, "score"):
                raise TypeError("Scorer must return ScoringResult or VariantReport.")
            name = scorer_columns[0]
            return {name: cls._scalar_score(result.score, scorer_name=name)}
        if set(report_results) != set(scorer_columns):
            raise ValueError(
                "ScorerSet result names do not match its configured names: "
                f"expected {tuple(scorer_columns)!r}, received {tuple(report_results)!r}."
            )
        return {
            name: cls._scalar_score(report_results[name].score, scorer_name=name)
            for name in scorer_columns
        }

    def score(
        self,
        interpreter: Any,
        scorer: Any,
        *,
        column_name: str | None = None,
        chunk_size: int = 128,
        grouping: str = "no_grouping",
        pair_execution: str = "separate",
        preprocessing_workers: int = 0,
        preprocessing_backend: str = "process",
        max_records_per_forward: int | None = None,
        max_pairs_per_forward: int | None = None,
        prefetch_batches: int = 1,
        show_progress: bool = True,
        overwrite: bool = False,
        on_error: Literal["raise", "record"] = "raise",
    ) -> "ISM":
        """Score pending mutations and add columns named after the scorer(s)."""

        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if on_error not in {"raise", "record"}:
            raise ValueError("on_error must be 'raise' or 'record'.")
        if not hasattr(interpreter, "score_sequence_pairs"):
            raise TypeError("interpreter must provide score_sequence_pairs(...).")

        pl = _require_polars()
        score_columns = self._scorer_columns(scorer, column_name)
        frame = self.variants
        additions = []
        for name in score_columns:
            if name not in frame.columns:
                additions.append(pl.lit(None, dtype=pl.Float64).alias(name))
            if on_error == "record" and f"{name}__error" not in frame.columns:
                additions.append(pl.lit(None, dtype=pl.Utf8).alias(f"{name}__error"))
        if additions:
            frame = frame.with_columns(additions)

        if overwrite:
            reset = [pl.lit(None, dtype=pl.Float64).alias(name) for name in score_columns]
            reset.extend(
                pl.lit(None, dtype=pl.Utf8).alias(f"{name}__error")
                for name in score_columns
                if f"{name}__error" in frame.columns
            )
            frame = frame.with_columns(reset)

        pending_indices = [
            index
            for index in range(frame.height)
            if any(frame.get_column(name)[index] is None for name in score_columns)
        ]
        self.variants = frame
        if not pending_indices:
            return self

        iterator: Any = range(0, len(pending_indices), chunk_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(
                    iterator,
                    total=math.ceil(len(pending_indices) / chunk_size),
                    desc=f"Scoring {', '.join(score_columns)}",
                    unit="chunk",
                )
            except ImportError:
                pass

        score_values = {name: frame.get_column(name).to_list() for name in score_columns}
        error_values = {
            name: (
                frame.get_column(f"{name}__error").to_list()
                if f"{name}__error" in frame.columns
                else None
            )
            for name in score_columns
        }

        for chunk_start in iterator:
            indices = pending_indices[chunk_start : chunk_start + chunk_size]
            rows = [frame.row(index, named=True) for index in indices]
            pairs = [self._sequence_pair_from_row(row) for row in rows]
            try:
                results = self._score_pairs(
                    interpreter,
                    scorer,
                    pairs,
                    grouping=grouping,
                    pair_execution=pair_execution,
                    preprocessing_workers=preprocessing_workers,
                    preprocessing_backend=preprocessing_backend,
                    max_records_per_forward=max_records_per_forward,
                    max_pairs_per_forward=max_pairs_per_forward,
                    prefetch_batches=prefetch_batches,
                    show_progress=False,
                )
                if len(results) != len(indices):
                    raise RuntimeError(
                        f"Interpreter returned {len(results)} results for {len(indices)} pairs."
                    )
                for index, result in zip(indices, results):
                    values = self._scores_from_result(result, score_columns)
                    for name, value in values.items():
                        score_values[name][index] = value
                        if error_values[name] is not None:
                            error_values[name][index] = None
            except Exception:
                if on_error == "raise":
                    raise
                for index, pair in zip(indices, pairs):
                    try:
                        result = self._score_pairs(
                            interpreter,
                            scorer,
                            [pair],
                            grouping=grouping,
                            pair_execution=pair_execution,
                            preprocessing_workers=preprocessing_workers,
                            preprocessing_backend=preprocessing_backend,
                            max_records_per_forward=max_records_per_forward,
                            max_pairs_per_forward=max_pairs_per_forward,
                            prefetch_batches=0,
                            show_progress=False,
                        )[0]
                        values = self._scores_from_result(result, score_columns)
                        for name, value in values.items():
                            score_values[name][index] = value
                            assert error_values[name] is not None
                            error_values[name][index] = None
                    except Exception as exc:
                        for name in score_columns:
                            assert error_values[name] is not None
                            error_values[name][index] = f"{type(exc).__name__}: {exc}"

            updates = [
                pl.Series(name, score_values[name], dtype=pl.Float64)
                for name in score_columns
            ]
            updates.extend(
                pl.Series(f"{name}__error", error_values[name], dtype=pl.Utf8)
                for name in score_columns
                if error_values[name] is not None
            )
            self.variants = self.variants.with_columns(updates)

        return self

    def _score_pairs(
        self,
        interpreter: Any,
        scorer: Any,
        pairs: Sequence[SequencePair],
        **kwargs: Any,
    ) -> list[Any]:
        return interpreter.score_sequence_pairs(
            pairs,
            condition=self.condition,
            scorer=scorer,
            center=_CENTER_FEATURE,
            retention="scalars",
            on_error="raise",
            **kwargs,
        )

    def pending(self, scorer_name: str):
        """Return variants whose requested scorer column is absent or null."""

        pl = _require_polars()
        if scorer_name not in self.variants.columns:
            return self.variants
        return self.variants.filter(pl.col(scorer_name).is_null())

    def failed(self, scorer_name: str):
        """Return variants with recorded errors for one scorer."""

        pl = _require_polars()
        error_column = f"{scorer_name}__error"
        if error_column not in self.variants.columns:
            return self.variants.head(0)
        return self.variants.filter(pl.col(error_column).is_not_null())

    def top(self, scorer_name: str, *, n: int = 20, absolute: bool = True):
        """Return the strongest finite scores for one scorer."""

        if scorer_name not in self.scorer_names:
            raise KeyError(f"Unknown scorer {scorer_name!r}; available: {self.scorer_names!r}.")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        pl = _require_polars()
        frame = self.variants.filter(
            pl.col(scorer_name).is_not_null() & pl.col(scorer_name).is_finite()
        )
        if absolute:
            return (
                frame.with_columns(pl.col(scorer_name).abs().alias("__absolute_score"))
                .sort("__absolute_score", descending=True)
                .drop("__absolute_score")
                .head(n)
            )
        return frame.sort(scorer_name, descending=True).head(n)

    def reset_scores(self, *scorer_names: str) -> "ISM":
        """Remove selected score and error columns, or all score columns."""

        names = tuple(scorer_names) or self.scorer_names
        columns = []
        for name in names:
            if name in self.variants.columns:
                columns.append(name)
            error_column = f"{name}__error"
            if error_column in self.variants.columns:
                columns.append(error_column)
        if columns:
            self.variants = self.variants.drop(columns)
        return self

    def summary(self) -> dict[str, Any]:
        """Return compact library and per-scorer completion counts."""

        pl = _require_polars()
        mutation_counts = {
            row["mutation_type"]: row["len"]
            for row in self.variants.group_by("mutation_type").len().to_dicts()
        }
        scorers = {}
        for name in self.scorer_names:
            scorers[name] = {
                "scored": self.variants.filter(pl.col(name).is_not_null()).height,
                "pending": self.variants.filter(pl.col(name).is_null()).height,
                "failed": self.failed(name).height,
            }
        return {
            "sequence_name": self.sequence.name,
            "sequence_length": len(self.sequence),
            "region": {"name": self.region.name, "start": self.region.start, "end": self.region.end},
            "center_position": self.center_position,
            "variants": self.variants.height,
            "mutation_counts": mutation_counts,
            "scorers": scorers,
        }

    def write_parquet(self, path: str | Path) -> Path:
        """Write the current mutation and score table to Parquet."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.variants.write_parquet(output)
        return output

    def plot(
        self,
        scorer_name: str,
        *,
        mutation_type: Literal["substitution", "deletion", "all"] = "substitution",
        target_col: str | None = None,
        position_mode: Literal["region", "sequence"] = "region",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot one scorer's effects as colored, reference-shaped lollipops."""

        from .plotting import plot_variant_effects

        return plot_variant_effects(
            self,
            scorer_name,
            mutation_type=mutation_type,
            target_col=target_col,
            position_mode=position_mode,
            figsize=figsize,
        )

    def plot_substitution_matrix(
        self,
        scorer_name: str,
        *,
        cmap: str = "coolwarm",
        center: float | None = 0.0,
        figsize: tuple[float, float] = (16.0, 4.0),
    ):
        """Plot alternative base by region position as a score heatmap."""

        from .plotting import plot_substitution_matrix

        return plot_substitution_matrix(
            self,
            scorer_name,
            cmap=cmap,
            center=center,
            figsize=figsize,
        )

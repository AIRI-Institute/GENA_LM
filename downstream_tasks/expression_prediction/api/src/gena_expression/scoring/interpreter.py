"""High-level variant and sequence-pair scoring orchestration."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..conditions import Condition
from ..config import FULL_RETENTION, RetentionLike, RetentionPolicy, retention_policy_for
from ..sequences import Feature, SequencePair
from ..variants import Variant
from ..inference.batching import (
    GroupingMode,
    PairExecutionMode,
    PreprocessingBackend,
    _progress,
)
from ..inference.model import SequenceModel
from .results import ScoringResult, VariantReport, retain_scoring_output


class VariantInterpreter:
    """High-level object for scoring variant and sequence-pair effects."""

    def __init__(self, model: SequenceModel) -> None:
        """Attach the sequence model used by every scoring workflow."""

        self.model = model
        self.logger = logging.getLogger(__name__)

    def score_variant(
        self,
        variant: Any,
        *,
        context: Any | None = None,
        condition: Condition | str | Mapping[str, Any],
        scorer: Any,
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        grouping: GroupingMode = "no_grouping",
        pair_execution: PairExecutionMode = "separate",
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
        retention: RetentionLike = None,
        **sequence_pair_kwargs: Any,
    ) -> ScoringResult | VariantReport:
        """Build, predict, and score one variant-derived sequence pair."""

        pair, label = self._pair_from_variant_record(
            variant,
            context=context,
            genome=genome,
            coordinate_system=coordinate_system,
            fallback="variant",
            **sequence_pair_kwargs,
        )
        return self.score_sequence_pair(
            pair,
            condition=condition,
            scorer=scorer,
            center=center,
            grouping=grouping,
            pair_execution=pair_execution,
            label=label,
            retention=retention,
        )

    def score_variants(
        self,
        variants: Iterable[Any],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        context: Any | None = None,
        condition: Condition | str | Mapping[str, Any] | None = None,
        scorer: Any,
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        grouping: GroupingMode = "no_grouping",
        pair_execution: PairExecutionMode = "separate",
        preprocessing_workers: int = 0,
        preprocessing_backend: PreprocessingBackend = "process",
        max_records_per_forward: int | None = None,
        max_pairs_per_forward: int | None = None,
        prefetch_batches: int = 1,
        show_progress: bool = True,
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
        on_error: Literal["raise", "warn", "skip", "exit"] = "raise",
        retention: RetentionLike = None,
        **sequence_pair_kwargs: Any,
    ) -> list[ScoringResult | VariantReport]:
        """Score variant records with separate or joint allele prediction."""

        record_list = list(variants)
        if not record_list:
            return []
        condition_list = self.model._broadcast_conditions(len(record_list), conditions, condition)
        retention_policy = retention_policy_for(retention)
        self._validate_on_error(on_error)

        if on_error in {"warn", "skip"}:
            results: list[ScoringResult | VariantReport] = []
            rows = enumerate(zip(record_list, condition_list))
            rows = _progress(
                rows,
                total=len(record_list),
                description="Scoring variants",
                enabled=show_progress,
                unit="variant",
            )
            for idx, (record, row_condition) in rows:
                try:
                    pair, label = self._pair_from_variant_record(
                        record,
                        context=context,
                        genome=genome,
                        coordinate_system=coordinate_system,
                        fallback=f"variant_{idx}",
                        **sequence_pair_kwargs,
                    )
                    results.extend(
                        self._score_pairs(
                            [pair],
                            [row_condition],
                            [label],
                            scorer=scorer,
                            center=center,
                            grouping=grouping,
                            pair_execution=pair_execution,
                            preprocessing_workers=preprocessing_workers,
                            preprocessing_backend=preprocessing_backend,
                            max_records_per_forward=max_records_per_forward,
                            max_pairs_per_forward=max_pairs_per_forward,
                            prefetch_batches=prefetch_batches,
                            show_progress=False,
                            retention=retention_policy,
                        )
                    )
                except Exception as exc:
                    self._handle_scoring_error(exc, item=record, on_error=on_error)
            return results

        try:
            pairs: list[SequencePair] = []
            labels: list[str] = []
            records_with_index = _progress(
                enumerate(record_list),
                total=len(record_list),
                description="Building variant pairs",
                enabled=show_progress,
                unit="variant",
            )
            for idx, record in records_with_index:
                pair, label = self._pair_from_variant_record(
                    record,
                    context=context,
                    genome=genome,
                    coordinate_system=coordinate_system,
                    fallback=f"variant_{idx}",
                    **sequence_pair_kwargs,
                )
                pairs.append(pair)
                labels.append(label)
            return self._score_pairs(
                pairs,
                condition_list,
                labels,
                scorer=scorer,
                center=center,
                grouping=grouping,
                pair_execution=pair_execution,
                preprocessing_workers=preprocessing_workers,
                preprocessing_backend=preprocessing_backend,
                max_records_per_forward=max_records_per_forward,
                max_pairs_per_forward=max_pairs_per_forward,
                prefetch_batches=prefetch_batches,
                show_progress=show_progress,
                retention=retention_policy,
            )
        except Exception as exc:
            self._handle_scoring_error(exc, item="variant batch", on_error=on_error)
            return []

    def score_sequence_pair(
        self,
        pair: SequencePair,
        *,
        condition: Condition | str | Mapping[str, Any],
        scorer: Any,
        center: int | str | Feature = "tss",
        grouping: GroupingMode = "no_grouping",
        pair_execution: PairExecutionMode = "separate",
        label: str | None = None,
        retention: RetentionLike = None,
    ) -> ScoringResult | VariantReport:
        """Predict and score one already materialized sequence pair."""

        return self._score_pairs(
            [pair],
            [self.model._as_condition(condition)],
            [label or self._pair_label(pair, 0)],
            scorer=scorer,
            center=center,
            grouping=grouping,
            pair_execution=pair_execution,
            prefetch_batches=0,
            show_progress=False,
            retention=retention_policy_for(retention),
        )[0]

    def score_sequence_pairs(
        self,
        pairs: Iterable[SequencePair],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        condition: Condition | str | Mapping[str, Any] | None = None,
        scorer: Any,
        center: int | str | Feature = "tss",
        grouping: GroupingMode = "no_grouping",
        pair_execution: PairExecutionMode = "separate",
        preprocessing_workers: int = 0,
        preprocessing_backend: PreprocessingBackend = "process",
        max_records_per_forward: int | None = None,
        max_pairs_per_forward: int | None = None,
        prefetch_batches: int = 1,
        show_progress: bool = True,
        on_error: Literal["raise", "warn", "skip", "exit"] = "raise",
        retention: RetentionLike = None,
    ) -> list[ScoringResult | VariantReport]:
        """Score sequence pairs with separate or joint allele prediction."""

        pair_list = list(pairs)
        if not pair_list:
            return []
        condition_list = self.model._broadcast_conditions(len(pair_list), conditions, condition)
        labels = [self._pair_label(pair, idx) for idx, pair in enumerate(pair_list)]
        retention_policy = retention_policy_for(retention)
        self._validate_on_error(on_error)

        if on_error in {"warn", "skip"}:
            results: list[ScoringResult | VariantReport] = []
            rows = zip(pair_list, condition_list, labels)
            rows = _progress(
                rows,
                total=len(pair_list),
                description="Scoring pairs",
                enabled=show_progress,
                unit="pair",
            )
            for pair, row_condition, label in rows:
                try:
                    results.extend(
                        self._score_pairs(
                            [pair],
                            [row_condition],
                            [label],
                            scorer=scorer,
                            center=center,
                            grouping=grouping,
                            pair_execution=pair_execution,
                            preprocessing_workers=preprocessing_workers,
                            preprocessing_backend=preprocessing_backend,
                            max_records_per_forward=max_records_per_forward,
                            max_pairs_per_forward=max_pairs_per_forward,
                            prefetch_batches=prefetch_batches,
                            show_progress=False,
                            retention=retention_policy,
                        )
                    )
                except Exception as exc:
                    self._handle_scoring_error(exc, item=label, on_error=on_error)
            return results

        try:
            return self._score_pairs(
                pair_list,
                condition_list,
                labels,
                scorer=scorer,
                center=center,
                grouping=grouping,
                pair_execution=pair_execution,
                preprocessing_workers=preprocessing_workers,
                preprocessing_backend=preprocessing_backend,
                max_records_per_forward=max_records_per_forward,
                max_pairs_per_forward=max_pairs_per_forward,
                prefetch_batches=prefetch_batches,
                show_progress=show_progress,
                retention=retention_policy,
            )
        except Exception as exc:
            self._handle_scoring_error(exc, item="sequence-pair batch", on_error=on_error)
            return []

    def _score_pairs(
        self,
        pairs: Sequence[SequencePair],
        conditions: Sequence[Condition],
        labels: Sequence[str],
        *,
        scorer: Any,
        center: int | str | Feature,
        grouping: GroupingMode,
        pair_execution: PairExecutionMode = "separate",
        preprocessing_workers: int = 0,
        preprocessing_backend: PreprocessingBackend = "process",
        max_records_per_forward: int | None = None,
        max_pairs_per_forward: int | None = None,
        prefetch_batches: int = 1,
        show_progress: bool = True,
        retention: RetentionPolicy | RetentionLike = None,
    ) -> list[ScoringResult | VariantReport]:
        """Predict a pair batch, then score its CPU predictions in order."""

        retention_policy = retention_policy_for(retention)
        # Scorers need the complete prediction. Compaction happens immediately
        # after each score, so numerical execution stays identical to full mode.
        predictions = self.model.predict_multiple_pairs(
            pairs,
            conditions=conditions,
            center=center,
            grouping=grouping,
            pair_execution=pair_execution,
            preprocessing_workers=preprocessing_workers,
            preprocessing_backend=preprocessing_backend,
            max_records_per_forward=max_records_per_forward,
            max_pairs_per_forward=max_pairs_per_forward,
            prefetch_batches=prefetch_batches,
            show_progress=show_progress,
            retention=FULL_RETENTION,
        )
        scorer_name = getattr(scorer, "name", type(scorer).__name__)
        self.logger.info("Scoring %d pair(s) with %s", len(predictions), scorer_name)
        rows = _progress(
            zip(range(len(predictions)), labels),
            total=len(predictions),
            description=f"Scoring {scorer_name}",
            enabled=show_progress,
            unit="pair",
        )
        results: list[ScoringResult | VariantReport] = []
        for index, label in rows:
            prediction = predictions[index]
            try:
                retained_scorer = getattr(scorer, "score_retained", None)
                scored = (
                    retained_scorer(prediction, retention_policy)
                    if callable(retained_scorer) and retention_policy != FULL_RETENTION
                    else scorer.score(prediction)
                )
                named = self._name_scoring_output(
                    scored,
                    label=label,
                )
                results.append(
                    retain_scoring_output(named, retention_policy)
                )
            finally:
                # Release each full prediction as soon as every scorer has used it.
                predictions[index] = None  # type: ignore[list-item]
        return results

    def _pair_from_variant_record(
        self,
        record: Any,
        *,
        context: Any | None,
        genome: Any | None,
        coordinate_system: Literal["auto", "0-based", "1-based"],
        fallback: str,
        **sequence_pair_kwargs: Any,
    ) -> tuple[SequencePair, str]:
        """Normalize a variant record and materialize its labeled pair."""

        variant = self._variant_from_record(record, genome=genome, coordinate_system=coordinate_system)
        explicit_name = sequence_pair_kwargs.pop("name", None)
        label = str(explicit_name or self._variant_record_name(record, variant=variant, fallback=fallback))
        pair = variant.to_sequence_pair(
            context=context,
            genome=genome,
            name=label,
            **sequence_pair_kwargs,
        )
        return pair, label

    @staticmethod
    def _variant_record_name(record: Any, *, variant: Variant, fallback: str) -> str:
        """Return a stable label for a variant record."""

        if variant.id:
            return str(variant.id)
        if isinstance(record, Mapping):
            return str(record.get("name") or record.get("id") or record.get("variant") or fallback)
        return str(getattr(record, "name", getattr(record, "id", fallback)) or fallback)

    @staticmethod
    def _variant_from_record(
        record: Any,
        *,
        genome: Any | None = None,
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
    ) -> Variant:
        """Normalize a variant object/string/table row to :class:`Variant`."""

        if isinstance(record, Variant):
            return record
        if isinstance(record, str):
            return Variant.from_str(record, genome=genome, coordinate_system=coordinate_system)
        if isinstance(record, Mapping):
            if "variant" in record:
                value = record["variant"]
                if isinstance(value, Variant):
                    return value
                return Variant.from_str(str(value), genome=genome, coordinate_system=coordinate_system)
            if {"chrom", "pos", "ref", "alt"}.issubset(record):
                variant = Variant(
                    chrom=str(record["chrom"]),
                    pos=int(record["pos"]) - 1 if coordinate_system == "1-based" else int(record["pos"]),
                    ref=str(record["ref"]),
                    alt=str(record["alt"]),
                    id=str(record.get("name") or record.get("id") or ""),
                )
                if genome is not None:
                    variant.validate(genome)
                return variant

        value = getattr(record, "variant", None)
        if value is not None:
            if isinstance(value, Variant):
                return value
            return Variant.from_str(str(value), genome=genome, coordinate_system=coordinate_system)
        chrom = getattr(record, "chrom", None)
        pos = getattr(record, "pos", None)
        ref = getattr(record, "ref", None)
        alt = getattr(record, "alt", None)
        if chrom is not None and pos is not None and ref is not None and alt is not None:
            variant = Variant(
                chrom=str(chrom),
                pos=int(pos) - 1 if coordinate_system == "1-based" else int(pos),
                ref=str(ref),
                alt=str(alt),
                id=str(getattr(record, "name", getattr(record, "id", "")) or ""),
            )
            if genome is not None:
                variant.validate(genome)
            return variant
        raise TypeError("Variant records must be Variant objects, strings, or rows with variant/chrom/pos/ref/alt fields.")

    @staticmethod
    def _pair_label(pair: SequencePair, index: int) -> str:
        """Choose a stable human-readable label for a sequence pair."""

        variant = pair.variant
        if hasattr(variant, "id") and variant.id:
            return str(variant.id)
        metadata = dict(pair.metadata or {})
        for key in ("name", "id", "variant_id"):
            if metadata.get(key):
                return str(metadata[key])
        if pair.ref.name and pair.ref.name == pair.alt.name:
            return pair.ref.name
        return f"pair_{index}"

    @staticmethod
    def _name_scoring_output(output: ScoringResult | VariantReport, *, label: str) -> ScoringResult | VariantReport:
        """Prefix one result or report with its input label."""

        if isinstance(output, ScoringResult):
            output.name = f"{label}:{output.name}"
            return output
        if isinstance(output, VariantReport):
            for key, result in output.results.items():
                scorer_name = result.name or key
                result.name = f"{label}:{scorer_name}"
            return output
        raise TypeError("scorer.score(...) must return ScoringResult or VariantReport.")

    @staticmethod
    def _validate_on_error(on_error: str) -> None:
        """Validate batch error-handling behavior."""

        if on_error not in {"raise", "warn", "skip", "exit"}:
            raise ValueError("on_error must be 'raise', 'warn', 'skip', or 'exit'.")

    @staticmethod
    def _handle_scoring_error(exc: Exception, *, item: Any, on_error: str) -> None:
        """Raise, exit, warn, or silently skip one scoring failure."""

        if on_error == "raise":
            raise exc
        if on_error == "exit":
            raise SystemExit(f"Scoring failed for {item}: {exc}") from exc
        if on_error == "warn":
            import warnings

            warnings.warn(f"Skipping {item}: {exc}", RuntimeWarning, stacklevel=2)

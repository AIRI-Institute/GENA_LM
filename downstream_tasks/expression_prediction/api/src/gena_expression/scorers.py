"""Scorers that turn reference/alternative predictions into variant effects."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence

from .predictions import PairPrediction
from .results import ScoreWindow, ScoringResult, VariantReport
from .sequences import Feature


class Scorer(Protocol):
    """Protocol for score objects."""

    name: str

    def score(self, prediction: PairPrediction) -> ScoringResult | VariantReport:
        """Score a reference/alternative prediction pair."""


class _TrackIntervalIndex:
    """Fast weighted interval queries over ordinary model-track rows.

    Model token rows are normally coordinate-sorted, non-overlapping, and
    finite. That common case uses prefix integrals. Unusual overlapping,
    unsorted, or non-finite rows retain the original scan semantics.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, value_key: str = "value") -> None:
        self.rows = [dict(row) for row in rows if int(row["end"]) > int(row["start"])]
        self.value_key = value_key
        self.starts = [int(row["start"]) for row in self.rows]
        self.ends = [int(row["end"]) for row in self.rows]
        self.values = [float(row[value_key]) for row in self.rows]
        self.fast = all(
            self.starts[index] >= self.ends[index - 1]
            for index in range(1, len(self.rows))
        ) and all(isfinite(value) for value in self.values)

        self.prefix_area = [0.0]
        self.prefix_coverage = [0]
        if self.fast:
            for start, end, value in zip(self.starts, self.ends, self.values):
                width = end - start
                self.prefix_area.append(self.prefix_area[-1] + value * width)
                self.prefix_coverage.append(self.prefix_coverage[-1] + width)

    def _scan_integral(self, start: int, end: int) -> tuple[float, int]:
        if end <= start:
            return 0.0, 0
        weighted_sum = 0.0
        total_weight = 0
        for row, value in zip(self.rows, self.values):
            overlap_start = max(int(row["start"]), start)
            overlap_end = min(int(row["end"]), end)
            if overlap_end <= overlap_start:
                continue
            weight = overlap_end - overlap_start
            weighted_sum += value * weight
            total_weight += weight
        return weighted_sum, total_weight

    def integral(self, start: int, end: int) -> tuple[float, int]:
        """Return weighted area and covered bases inside ``[start, end)``."""

        start = int(start)
        end = int(end)
        if end <= start:
            return 0.0, 0
        if not self.fast:
            return self._scan_integral(start, end)

        left = bisect_right(self.ends, start)
        right = bisect_left(self.starts, end)
        if right <= left:
            return 0.0, 0

        weighted_sum = self.prefix_area[right] - self.prefix_area[left]
        total_weight = self.prefix_coverage[right] - self.prefix_coverage[left]
        first = left
        last = right - 1
        left_clip = max(0, start - self.starts[first])
        right_clip = max(0, self.ends[last] - end)
        if left_clip:
            weighted_sum -= self.values[first] * left_clip
            total_weight -= left_clip
        if right_clip:
            weighted_sum -= self.values[last] * right_clip
            total_weight -= right_clip
        return weighted_sum, max(0, total_weight)

    def integral_many(self, windows: Sequence[ScoreWindow]) -> tuple[float, int]:
        """Return combined weighted area and coverage for supplied windows."""

        weighted_sum = 0.0
        total_weight = 0
        for window in windows:
            window_sum, window_weight = self.integral(window.start, window.end)
            weighted_sum += window_sum
            total_weight += window_weight
        return weighted_sum, total_weight

    def prefix_at_sorted(self, positions: Sequence[int]) -> tuple[list[float], list[int]] | None:
        """Evaluate cumulative area/coverage at nondecreasing coordinates in O(N+T)."""

        if not self.fast or any(positions[index] < positions[index - 1] for index in range(1, len(positions))):
            return None
        areas: list[float] = []
        coverages: list[int] = []
        row_index = 0
        completed_area = 0.0
        completed_coverage = 0
        for position in positions:
            position = int(position)
            while row_index < len(self.rows) and self.ends[row_index] <= position:
                width = self.ends[row_index] - self.starts[row_index]
                completed_area += self.values[row_index] * width
                completed_coverage += width
                row_index += 1
            area = completed_area
            coverage = completed_coverage
            if (
                row_index < len(self.rows)
                and self.starts[row_index] < position < self.ends[row_index]
            ):
                width = position - self.starts[row_index]
                area += self.values[row_index] * width
                coverage += width
            areas.append(area)
            coverages.append(coverage)
        return areas, coverages

    def weighted_means_for_sorted_intervals(
        self,
        intervals: Sequence[tuple[int, int]],
    ) -> list[float | None] | None:
        """Return overlap-weighted means using two linear prefix sweeps."""

        starts = [int(start) for start, _ in intervals]
        ends = [int(end) for _, end in intervals]
        start_prefix = self.prefix_at_sorted(starts)
        end_prefix = self.prefix_at_sorted(ends)
        if start_prefix is None or end_prefix is None:
            return None
        start_areas, start_coverages = start_prefix
        end_areas, end_coverages = end_prefix
        means: list[float | None] = []
        for start, end, start_area, end_area, start_coverage, end_coverage in zip(
            starts,
            ends,
            start_areas,
            end_areas,
            start_coverages,
            end_coverages,
        ):
            if end <= start:
                means.append(None)
                continue
            coverage = end_coverage - start_coverage
            means.append((end_area - start_area) / coverage if coverage > 0 else None)
        return means


@dataclass(frozen=True)
class ExpressionDeltaScorer:
    """Score alternative minus reference expression."""

    output: str = "expression"
    sign: Literal["alt-ref", "ref-alt"] = "alt-ref"
    transform: Literal["none", "log2", "zscore"] = "none"
    name: str = "expression_delta"

    def _transform(self, value: float) -> float:
        if self.transform == "none":
            return value
        if self.transform == "log2":
            import math

            return math.log2(max(value, 0.0) + 1.0)
        raise ValueError("zscore transform requires a fitted reference distribution.")

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Return an expression effect score."""

        ref = prediction.ref.scalar(self.output)
        alt = prediction.alt.scalar(self.output)
        ref_values = ref if isinstance(ref, list) else [ref]
        alt_values = alt if isinstance(alt, list) else [alt]
        n = min(len(ref_values), len(alt_values))
        deltas = [self._transform(float(alt_values[i])) - self._transform(float(ref_values[i])) for i in range(n)]
        if self.sign == "ref-alt":
            deltas = [-value for value in deltas]
        score = deltas[0] if len(deltas) == 1 else deltas
        return ScoringResult(
            name=self.name,
            score=score,
            prediction=prediction,
            provenance={"scorer": type(self).__name__, "output": self.output, "sign": self.sign},
        )


@dataclass(frozen=True)
class TokenWindowScorer:
    """Score directly over model token positions.

    This preserves the attached ATAC scoring behavior:

    ``alt_logits[:, start:end, :].sum(dim=1) - wt_logits[:, start:end, :].sum(dim=1)``
    """

    output: str = "atac"
    tokens_left: int = 37
    tokens_right: int = 37
    center_token: Literal["middle", "variant", "tss"] = "middle"
    aggregate: Literal["sum", "mean", "max", "max_abs_delta"] = "sum"
    sign: Literal["alt-ref", "ref-alt"] = "alt-ref"
    channel: int | None = 0
    name: str = "token_window"

    def _select_channel(self, logits: Any):
        if logits.ndim == 3:
            if logits.shape[-1] == 1:
                return logits[:, :, 0]
            if self.channel is None:
                return logits.mean(dim=-1)
            return logits[:, :, int(self.channel)]
        if logits.ndim == 2:
            return logits
        raise ValueError(f"Expected 2D or 3D logits, got shape {tuple(logits.shape)}")

    def _aggregate(self, values: Any):
        if self.aggregate == "sum":
            return values.sum(dim=1)
        if self.aggregate == "mean":
            return values.mean(dim=1)
        if self.aggregate == "max":
            return values.max(dim=1).values
        raise ValueError("max_abs_delta is applied after delta calculation.")

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Score a token window around the common middle point."""

        wt_logits = prediction.ref.logits
        alt_logits = prediction.alt.logits
        wt_num_tokens = wt_logits.shape[1]
        alt_num_tokens = alt_logits.shape[1]
        common_middle_point = ((wt_num_tokens // 2) + (alt_num_tokens // 2)) // 2

        start = common_middle_point - self.tokens_left
        end = common_middle_point + self.tokens_right
        if start < 0:
            start = 0
        if end > min(wt_num_tokens, alt_num_tokens):
            end = min(wt_num_tokens, alt_num_tokens)
        if start < 0 or end < 0 or end <= start:
            raise ValueError(f"Invalid token scoring window: start={start}, end={end}")

        wt_slice = wt_logits[:, start:end, :] if wt_logits.ndim == 3 else wt_logits[:, start:end]
        alt_slice = alt_logits[:, start:end, :] if alt_logits.ndim == 3 else alt_logits[:, start:end]
        wt_values = self._select_channel(wt_slice)
        alt_values = self._select_channel(alt_slice)
        if self.aggregate == "max_abs_delta":
            score_tensor = (alt_values - wt_values).abs().max(dim=1).values
        else:
            score_tensor = self._aggregate(alt_values) - self._aggregate(wt_values)
        if self.sign == "ref-alt":
            score_tensor = -score_tensor

        values = score_tensor.detach().float().cpu().reshape(-1).tolist()
        score = float(values[0]) if len(values) == 1 else [float(value) for value in values]
        return ScoringResult(
            name=self.name,
            score=score,
            prediction=prediction,
            score_window=ScoreWindow(start=start, end=end, center=common_middle_point, units="token"),
            provenance={
                "scorer": type(self).__name__,
                "output": self.output,
                "aggregate": self.aggregate,
                "sign": self.sign,
                "copied_behavior": "attached CAGI5_bench atac_seq token window",
            },
        )


@dataclass(frozen=True)
class TrackWindowScorer:
    """Score a coordinate-aware token track inside a sequence window."""

    track: str = "atac"
    center: str | int | Feature = "variant"
    width_bp: int | None = None
    left_bp: int | None = None
    right_bp: int | None = None
    aggregate: Literal["sum", "mean", "max", "min", "max_abs_delta", "auc", "weighted_sum"] = "sum"
    sign: Literal["alt-ref", "ref-alt"] = "alt-ref"
    channel: int | None = 0
    name: str = "track_window"

    def __post_init__(self) -> None:
        if self.width_bp is None and (self.left_bp is None or self.right_bp is None):
            object.__setattr__(self, "width_bp", 501)
        if self.width_bp is not None and (self.left_bp is not None or self.right_bp is not None):
            raise ValueError("Use either width_bp or left_bp/right_bp, not both.")

    def _resolve_center(self, prediction: PairPrediction) -> int:
        if isinstance(self.center, int):
            return self.center
        if isinstance(self.center, Feature):
            return (self.center.start + self.center.end) // 2
        if self.center == "variant":
            feature = prediction.pair.variant_feature()
            if feature is None:
                raise ValueError("No variant feature is available for TrackWindowScorer.")
            return (feature.start + feature.end) // 2
        center = self.center
        if isinstance(center, str) and center.startswith("feature:"):
            center = center.split(":", 1)[1]
        feature = prediction.pair.ref.feature(str(center), required=True)
        assert feature is not None
        return (feature.start + feature.end) // 2

    def window_for(self, prediction: PairPrediction) -> ScoreWindow:
        """Return the score window for a prediction pair."""

        center = self._resolve_center(prediction)
        if self.width_bp is not None:
            start = center - int(self.width_bp) // 2
            end = start + int(self.width_bp)
        else:
            start = center - int(self.left_bp)
            end = center + int(self.right_bp)
        return ScoreWindow(start=start, end=end, center=center, units="sequence")

    @staticmethod
    def _rows_with_values(track_prediction) -> list[dict[str, Any]]:
        return [{**row, "value": value} for row, value in zip(track_prediction.tokens, track_prediction.values)]

    def _aggregate_rows(
        self,
        rows: list[dict[str, Any]],
        window: ScoreWindow,
        *,
        index: _TrackIntervalIndex | None = None,
    ) -> float:
        if self.aggregate in {"sum", "auc", "weighted_sum", "mean"}:
            index = index or _TrackIntervalIndex(rows)
            weighted_sum, total_weight = index.integral(window.start, window.end)
            if total_weight <= 0:
                return float("nan")
            if self.aggregate == "mean":
                return float(weighted_sum / total_weight)
            return float(weighted_sum)

        values: list[float] = []
        for row in rows:
            overlap_start = max(int(row["start"]), window.start)
            overlap_end = min(int(row["end"]), window.end)
            if overlap_end <= overlap_start:
                continue
            values.append(float(row["value"]))
        if not values:
            return float("nan")
        if self.aggregate == "max":
            return float(max(values))
        if self.aggregate == "min":
            return float(min(values))
        raise ValueError("max_abs_delta is computed from ref and alt rows together.")

    @staticmethod
    def _mean_over_interval(rows: list[dict[str, Any]], start: int, end: int) -> float | None:
        """Return an overlap-weighted mean track value for one interval."""

        if end <= start:
            return None
        weighted_sum = 0.0
        total_weight = 0
        for row in rows:
            overlap_start = max(int(row["start"]), start)
            overlap_end = min(int(row["end"]), end)
            if overlap_end <= overlap_start:
                continue
            weight = overlap_end - overlap_start
            weighted_sum += float(row["value"]) * weight
            total_weight += weight
        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    @classmethod
    def _delta_rows(
        cls,
        ref_rows: list[dict[str, Any]],
        alt_rows: list[dict[str, Any]],
        sign: str,
        *,
        pair: Any | None = None,
        mapper: Any | None = None,
        alt_index: _TrackIntervalIndex | None = None,
    ) -> list[dict[str, Any]]:
        """Return delta rows in reference coordinates.

        When a sequence pair is supplied, every reference token interval is
        mapped into alternative-local coordinates before alternative values are
        aggregated. Deleted or otherwise uncovered intervals stay masked as
        ``nan``; no synthetic zero or average value is inserted.
        """

        rows: list[dict[str, Any]] = []
        multiplier = -1.0 if sign == "ref-alt" else 1.0
        if pair is None:
            n = min(len(ref_rows), len(alt_rows))
            for idx in range(n):
                row = dict(alt_rows[idx])
                row["ref_value"] = ref_rows[idx]["value"]
                row["alt_value"] = alt_rows[idx]["value"]
                row["delta"] = multiplier * (alt_rows[idx]["value"] - ref_rows[idx]["value"])
                row["alignment_status"] = "index"
                rows.append(row)
            return rows

        mapper = mapper or pair.coordinate_mapper()
        alt_index = alt_index or _TrackIntervalIndex(alt_rows)
        mapped_intervals = [
            mapper.map_ref_interval_to_alt(int(ref_row["start"]), int(ref_row["end"]))
            for ref_row in ref_rows
        ]
        alt_values = alt_index.weighted_means_for_sorted_intervals(mapped_intervals)
        if alt_values is None:
            alt_values = [
                cls._mean_over_interval(alt_rows, alt_start, alt_end)
                for alt_start, alt_end in mapped_intervals
            ]

        for ref_row, (alt_start, alt_end), alt_value in zip(ref_rows, mapped_intervals, alt_values):
            row = dict(ref_row)
            ref_value = float(ref_row["value"])
            row["ref_value"] = ref_value
            row["alt_start"] = alt_start
            row["alt_end"] = alt_end
            if alt_value is None:
                row["alt_value"] = float("nan")
                row["delta"] = float("nan")
                row["alignment_status"] = "deleted" if alt_end <= alt_start else "uncovered"
            else:
                row["alt_value"] = float(alt_value)
                row["delta"] = multiplier * (float(alt_value) - ref_value)
                row["alignment_status"] = "mapped"
            rows.append(row)
        return rows

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Return a windowed track-effect score."""

        window = self.window_for(prediction)
        ref_track = prediction.ref.track(self.track, channel=self.channel)
        alt_track = prediction.alt.track(self.track, channel=self.channel)
        ref_rows = self._rows_with_values(ref_track)
        alt_rows = self._rows_with_values(alt_track)
        mapper = prediction.pair.coordinate_mapper()
        ref_index = _TrackIntervalIndex(ref_rows)
        alt_index = _TrackIntervalIndex(alt_rows)
        alt_start, alt_end = mapper.map_ref_interval_to_alt(window.start, window.end)
        alt_window = ScoreWindow(
            start=alt_start,
            end=alt_end,
            center=(alt_start + alt_end) // 2,
            units="sequence",
        )
        delta_rows = self._delta_rows(
            ref_rows,
            alt_rows,
            self.sign,
            pair=prediction.pair,
            mapper=mapper,
            alt_index=alt_index,
        )

        if self.aggregate == "max_abs_delta":
            selected = [
                abs(float(row["delta"]))
                for row in delta_rows
                if max(int(row["start"]), window.start) < min(int(row["end"]), window.end)
                and isfinite(float(row["delta"]))
            ]
            score = max(selected) if selected else float("nan")
        else:
            ref_score = self._aggregate_rows(ref_rows, window, index=ref_index)
            alt_score = self._aggregate_rows(alt_rows, alt_window, index=alt_index)
            score = alt_score - ref_score
            if self.sign == "ref-alt":
                score = -score

        return ScoringResult(
            name=self.name,
            score=float(score),
            prediction=prediction,
            score_window=window,
            ref_track=ref_rows,
            alt_track=alt_rows,
            delta_track=delta_rows,
            provenance={
                "scorer": type(self).__name__,
                "track": self.track,
                "aggregate": self.aggregate,
                "sign": self.sign,
                "channel": self.channel,
                "ref_window": window.to_dict(),
                "alt_window": alt_window.to_dict(),
            },
        )


@dataclass(frozen=True)
class TrackFeatureScorer:
    """Score a coordinate-aware track over annotated sequence features.

    This scorer behaves like :class:`TrackWindowScorer`, but instead of using a
    fixed window around a center, it aggregates only track bins overlapping one
    or more named features on the reference and alternative sequences.
    """

    features: str | Feature | Sequence[str | Feature]
    track: str = "atac"
    aggregate: Literal["sum", "mean", "max", "min", "max_abs_delta", "auc", "weighted_sum"] = "sum"
    sign: Literal["alt-ref", "ref-alt"] = "alt-ref"
    channel: int | None = 0
    name: str = "track_feature"

    def _feature_specs(self) -> tuple[str | Feature, ...]:
        """Return feature selectors as a tuple."""

        if isinstance(self.features, (str, Feature)):
            return (self.features,)
        return tuple(self.features)

    @staticmethod
    def _matching_features(sequence: Any, selector: str | Feature) -> list[Feature]:
        """Return all sequence features matching one selector."""

        if isinstance(selector, Feature):
            return [selector]
        name = selector.split(":", 1)[1] if selector.startswith("feature:") else selector
        return [feature for feature in sequence.features if feature.name == name or feature.type == name]

    @classmethod
    def _feature_windows(cls, sequence: Any, selectors: Sequence[str | Feature]) -> list[ScoreWindow]:
        """Resolve feature selectors against one annotated sequence."""

        windows: list[ScoreWindow] = []
        missing: list[str] = []
        for selector in selectors:
            features = cls._matching_features(sequence, selector)
            if not features:
                missing.append(selector.name if isinstance(selector, Feature) else str(selector))
                continue
            windows.extend(
                ScoreWindow(
                    start=feature.start,
                    end=feature.end,
                    center=(feature.start + feature.end) // 2,
                    units="sequence",
                )
                for feature in features
            )
        if missing:
            raise ValueError(f"Features not found on sequence {sequence.name!r}: {', '.join(missing)}")
        return cls._merge_windows(windows)

    @staticmethod
    def _merge_windows(windows: Sequence[ScoreWindow]) -> list[ScoreWindow]:
        """Merge overlapping feature windows so track bins are not double-counted."""

        if not windows:
            return []
        ordered = sorted(windows, key=lambda window: (window.start, window.end))
        merged: list[ScoreWindow] = []
        current_start = ordered[0].start
        current_end = ordered[0].end
        for window in ordered[1:]:
            if window.start <= current_end:
                current_end = max(current_end, window.end)
            else:
                merged.append(
                    ScoreWindow(
                        start=current_start,
                        end=current_end,
                        center=(current_start + current_end) // 2,
                        units="sequence",
                    )
                )
                current_start = window.start
                current_end = window.end
        merged.append(
            ScoreWindow(
                start=current_start,
                end=current_end,
                center=(current_start + current_end) // 2,
                units="sequence",
            )
        )
        return merged

    @staticmethod
    def _bounding_window(windows: Sequence[ScoreWindow]) -> ScoreWindow | None:
        """Return one broad window for plotting/provenance."""

        if not windows:
            return None
        start = min(window.start for window in windows)
        end = max(window.end for window in windows)
        return ScoreWindow(start=start, end=end, center=(start + end) // 2, units="sequence")

    @staticmethod
    def _row_overlap_weight(row: Mapping[str, Any], windows: Sequence[ScoreWindow]) -> int:
        """Return how many bases of a track row overlap selected windows."""

        row_start = int(row["start"])
        row_end = int(row["end"])
        return sum(max(0, min(row_end, window.end) - max(row_start, window.start)) for window in windows)

    def _aggregate_rows_in_windows(
        self,
        rows: list[dict[str, Any]],
        windows: Sequence[ScoreWindow],
        *,
        index: _TrackIntervalIndex | None = None,
    ) -> float:
        """Aggregate track values over one or more feature windows."""

        if self.aggregate in {"sum", "auc", "weighted_sum", "mean"}:
            index = index or _TrackIntervalIndex(rows)
            weighted_sum, total_weight = index.integral_many(windows)
            if total_weight <= 0:
                return float("nan")
            if self.aggregate == "mean":
                return float(weighted_sum / total_weight)
            return float(weighted_sum)

        values: list[float] = []
        for row in rows:
            weight = self._row_overlap_weight(row, windows)
            if weight <= 0:
                continue
            values.append(float(row["value"]))
        if not values:
            return float("nan")
        if self.aggregate == "max":
            return float(max(values))
        if self.aggregate == "min":
            return float(min(values))
        raise ValueError("max_abs_delta is computed from ref and alt rows together.")

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Return a feature-restricted track-effect score."""

        selectors = self._feature_specs()
        ref_windows = self._feature_windows(prediction.pair.ref, selectors)
        mapper = prediction.pair.coordinate_mapper()
        alt_windows = self._merge_windows(
            [
                ScoreWindow(
                    start=mapped_start,
                    end=mapped_end,
                    center=(mapped_start + mapped_end) // 2,
                    units="sequence",
                )
                for window in ref_windows
                for mapped_start, mapped_end in [
                    mapper.map_ref_interval_to_alt(window.start, window.end)
                ]
            ]
        )
        ref_track = prediction.ref.track(self.track, channel=self.channel)
        alt_track = prediction.alt.track(self.track, channel=self.channel)
        ref_rows = TrackWindowScorer._rows_with_values(ref_track)
        alt_rows = TrackWindowScorer._rows_with_values(alt_track)
        ref_index = _TrackIntervalIndex(ref_rows)
        alt_index = _TrackIntervalIndex(alt_rows)
        delta_rows = TrackWindowScorer._delta_rows(
            ref_rows,
            alt_rows,
            self.sign,
            pair=prediction.pair,
            mapper=mapper,
            alt_index=alt_index,
        )

        if self.aggregate == "max_abs_delta":
            selected = [
                abs(float(row["delta"]))
                for row in delta_rows
                if self._row_overlap_weight(row, ref_windows) > 0
                and isfinite(float(row["delta"]))
            ]
            score = max(selected) if selected else float("nan")
        else:
            ref_score = self._aggregate_rows_in_windows(ref_rows, ref_windows, index=ref_index)
            alt_score = self._aggregate_rows_in_windows(alt_rows, alt_windows, index=alt_index)
            score = alt_score - ref_score
            if self.sign == "ref-alt":
                score = -score

        return ScoringResult(
            name=self.name,
            score=float(score),
            prediction=prediction,
            score_window=self._bounding_window((*ref_windows, *alt_windows)),
            ref_track=ref_rows,
            alt_track=alt_rows,
            delta_track=delta_rows,
            provenance={
                "scorer": type(self).__name__,
                "track": self.track,
                "features": [
                    selector.name if isinstance(selector, Feature) else str(selector)
                    for selector in selectors
                ],
                "ref_windows": [window.to_dict() for window in ref_windows],
                "alt_windows": [window.to_dict() for window in alt_windows],
                "aggregate": self.aggregate,
                "sign": self.sign,
                "channel": self.channel,
            },
        )


@dataclass(frozen=True)
class TrackAllFeaturesScorer:
    """Score every annotated feature on each prediction pair independently.

    Feature intervals are taken from ``feature_source`` and mapped through the
    pair's reference/alternative replacement before each native track is
    aggregated. The returned :class:`ScoringResult` stores the requested
    ``{feature: score}`` mapping directly in ``result.score``.
    """

    track: str = "atac"
    aggregate: Literal["sum", "mean", "max", "min", "max_abs_delta", "auc", "weighted_sum"] = "sum"
    sign: Literal["alt-ref", "ref-alt"] = "alt-ref"
    channel: int | None = 0
    feature_source: Literal["ref", "alt"] = "ref"
    name: str = "track_all_features"

    def __post_init__(self) -> None:
        if self.feature_source not in {"ref", "alt"}:
            raise ValueError("feature_source must be 'ref' or 'alt'.")

    @staticmethod
    def _feature_keys(features: Sequence[Feature]) -> list[str]:
        """Return stable, unique dictionary keys for sequence features."""

        bases = [feature.name or feature.type or "feature" for feature in features]
        totals: dict[str, int] = {}
        for base in bases:
            totals[base] = totals.get(base, 0) + 1
        occurrences: dict[str, int] = {}
        used: set[str] = set()
        keys: list[str] = []
        for base in bases:
            occurrences[base] = occurrences.get(base, 0) + 1
            key = base if totals[base] == 1 else f"{base}#{occurrences[base]}"
            if key in used:
                suffix = 2
                candidate = f"{key}#{suffix}"
                while candidate in used:
                    suffix += 1
                    candidate = f"{key}#{suffix}"
                key = candidate
            used.add(key)
            keys.append(key)
        return keys

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Return all per-feature scores for this prediction pair."""

        sequence = prediction.pair.ref if self.feature_source == "ref" else prediction.pair.alt
        features = tuple(sequence.features)
        ref_track = prediction.ref.track(self.track, channel=self.channel)
        alt_track = prediction.alt.track(self.track, channel=self.channel)
        ref_rows = TrackWindowScorer._rows_with_values(ref_track)
        alt_rows = TrackWindowScorer._rows_with_values(alt_track)
        mapper = prediction.pair.coordinate_mapper()
        ref_index = _TrackIntervalIndex(ref_rows)
        alt_index = _TrackIntervalIndex(alt_rows)
        delta_rows = TrackWindowScorer._delta_rows(
            ref_rows,
            alt_rows,
            self.sign,
            pair=prediction.pair,
            mapper=mapper,
            alt_index=alt_index,
        )
        aggregator = TrackFeatureScorer(
            features=(),
            track=self.track,
            aggregate=self.aggregate,
            sign=self.sign,
            channel=self.channel,
        )

        scores: dict[str, float] = {}
        feature_results: list[dict[str, Any]] = []
        ref_result_windows: list[ScoreWindow] = []
        for feature_key, feature in zip(self._feature_keys(features), features):
            source_window = ScoreWindow(
                start=feature.start,
                end=feature.end,
                center=(feature.start + feature.end) // 2,
                units="sequence",
            )
            if self.feature_source == "ref":
                ref_window = source_window
                alt_start, alt_end = mapper.map_ref_interval_to_alt(feature.start, feature.end)
                alt_window = ScoreWindow(
                    start=alt_start,
                    end=alt_end,
                    center=(alt_start + alt_end) // 2,
                    units="sequence",
                )
            else:
                alt_window = source_window
                ref_start, ref_end = mapper.map_alt_interval_to_ref(feature.start, feature.end)
                ref_window = ScoreWindow(
                    start=ref_start,
                    end=ref_end,
                    center=(ref_start + ref_end) // 2,
                    units="sequence",
                )
            ref_result_windows.append(ref_window)
            if self.aggregate == "max_abs_delta":
                selected = [
                    abs(float(row["delta"]))
                    for row in delta_rows
                    if aggregator._row_overlap_weight(row, (ref_window,)) > 0
                    and isfinite(float(row["delta"]))
                ]
                score = max(selected) if selected else float("nan")
            else:
                ref_score = aggregator._aggregate_rows_in_windows(
                    ref_rows,
                    (ref_window,),
                    index=ref_index,
                )
                alt_score = aggregator._aggregate_rows_in_windows(
                    alt_rows,
                    (alt_window,),
                    index=alt_index,
                )
                score = alt_score - ref_score
                if self.sign == "ref-alt":
                    score = -score

            scores[feature_key] = float(score)
            feature_results.append(
                {
                    "key": feature_key,
                    "score": float(score),
                    "feature": feature.to_dict(),
                    "score_window": ref_window.to_dict(),
                    "ref_score_window": ref_window.to_dict(),
                    "alt_score_window": alt_window.to_dict(),
                }
            )

        return ScoringResult(
            name=self.name,
            score=scores,
            prediction=prediction,
            score_window=TrackFeatureScorer._bounding_window(ref_result_windows),
            ref_track=ref_rows,
            alt_track=alt_rows,
            delta_track=delta_rows,
            features=feature_results,
            provenance={
                "scorer": type(self).__name__,
                "track": self.track,
                "feature_source": self.feature_source,
                "aggregate": self.aggregate,
                "sign": self.sign,
                "channel": self.channel,
            },
        )


@dataclass(frozen=True)
class TrackFeatureBuilder:
    """Build binned delta-track features for regression scorers."""

    track: str = "atac"
    center: str | int | Feature = "variant"
    width_bp: int = 5000
    bin_size: int = 50
    aggregate: Literal["mean", "sum"] = "mean"
    channel: int | None = 0

    def build(self, prediction: PairPrediction) -> tuple[list[float], list[dict[str, Any]]]:
        """Return feature vector and feature metadata."""

        scorer = TrackWindowScorer(
            track=self.track,
            center=self.center,
            width_bp=self.width_bp,
            aggregate="sum",
            channel=self.channel,
        )
        window = scorer.window_for(prediction)
        delta = prediction.delta_track(self.track, channel=self.channel)
        rows = [{**row, "delta": value} for row, value in zip(delta.tokens, delta.values)]
        features: list[float] = []
        metadata: list[dict[str, Any]] = []
        for start in range(window.start, window.end, self.bin_size):
            end = min(start + self.bin_size, window.end)
            values = [
                float(row["delta"])
                for row in rows
                if max(int(row["start"]), start) < min(int(row["end"]), end)
            ]
            if not values:
                feature_value = 0.0
            elif self.aggregate == "sum":
                feature_value = float(sum(values))
            else:
                feature_value = float(sum(values) / len(values))
            features.append(feature_value)
            metadata.append({"feature_start": start, "feature_end": end, "feature_center": (start + end) / 2})
        return features, metadata


@dataclass
class RegressionScorer:
    """Apply a trained regression model to features extracted from predictions."""

    model: Any
    feature_builder: TrackFeatureBuilder
    name: str = "regression_score"

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        feature_builder: TrackFeatureBuilder,
        name: str = "regression_score",
    ) -> "RegressionScorer":
        """Load a joblib regression model or bundle."""

        import joblib

        bundle = joblib.load(path)
        model = bundle["model"] if isinstance(bundle, Mapping) and "model" in bundle else bundle
        return cls(model=model, feature_builder=feature_builder, name=name)

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Return the regression score for one prediction pair."""

        import numpy as np

        features, metadata = self.feature_builder.build(prediction)
        X = np.asarray(features, dtype=float).reshape(1, -1)
        value = self.model.predict(X)
        score = float(np.asarray(value).reshape(-1)[0])
        return ScoringResult(
            name=self.name,
            score=score,
            prediction=prediction,
            features=metadata,
            provenance={"scorer": type(self).__name__},
        )


@dataclass(frozen=True)
class ScorerSet:
    """Run several scorers and return one report."""

    scorers: Mapping[str, Scorer] | Sequence[Scorer]

    def _items(self) -> Iterable[tuple[str, Scorer]]:
        if isinstance(self.scorers, Mapping):
            return self.scorers.items()
        return ((scorer.name, scorer) for scorer in self.scorers)

    def score(self, prediction: PairPrediction) -> VariantReport:
        """Score a prediction pair with every scorer."""

        return VariantReport({name: scorer.score(prediction) for name, scorer in self._items()})

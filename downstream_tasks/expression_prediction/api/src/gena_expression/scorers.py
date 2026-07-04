"""Scorers that turn reference/alternative predictions into variant effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence

from .predictions import PairPrediction
from .results import ScoreWindow, ScoringResult, VariantReport
from .sequences import Feature


class Scorer(Protocol):
    """Protocol for score objects."""

    name: str

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Score a reference/alternative prediction pair."""


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

    def _aggregate_rows(self, rows: list[dict[str, Any]], window: ScoreWindow) -> float:
        values: list[float] = []
        weights: list[int] = []
        for row in rows:
            overlap_start = max(int(row["start"]), window.start)
            overlap_end = min(int(row["end"]), window.end)
            if overlap_end <= overlap_start:
                continue
            values.append(float(row["value"]))
            weights.append(overlap_end - overlap_start)
        if not values:
            return float("nan")
        if self.aggregate in {"sum", "auc", "weighted_sum"}:
            return float(sum(value * weight for value, weight in zip(values, weights)))
        if self.aggregate == "mean":
            denominator = sum(weights)
            return float(sum(value * weight for value, weight in zip(values, weights)) / denominator)
        if self.aggregate == "max":
            return float(max(values))
        if self.aggregate == "min":
            return float(min(values))
        raise ValueError("max_abs_delta is computed from ref and alt rows together.")

    @staticmethod
    def _delta_rows(ref_rows: list[dict[str, Any]], alt_rows: list[dict[str, Any]], sign: str) -> list[dict[str, Any]]:
        n = min(len(ref_rows), len(alt_rows))
        rows = []
        multiplier = -1.0 if sign == "ref-alt" else 1.0
        for idx in range(n):
            row = dict(alt_rows[idx])
            row["ref_value"] = ref_rows[idx]["value"]
            row["alt_value"] = alt_rows[idx]["value"]
            row["delta"] = multiplier * (alt_rows[idx]["value"] - ref_rows[idx]["value"])
            rows.append(row)
        return rows

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Return a windowed track-effect score."""

        window = self.window_for(prediction)
        ref_track = prediction.ref.track(self.track, channel=self.channel)
        alt_track = prediction.alt.track(self.track, channel=self.channel)
        ref_rows = self._rows_with_values(ref_track)
        alt_rows = self._rows_with_values(alt_track)
        delta_rows = self._delta_rows(ref_rows, alt_rows, self.sign)

        if self.aggregate == "max_abs_delta":
            selected = [
                abs(float(row["delta"]))
                for row in delta_rows
                if max(int(row["start"]), window.start) < min(int(row["end"]), window.end)
            ]
            score = max(selected) if selected else float("nan")
        else:
            ref_score = self._aggregate_rows(ref_rows, window)
            alt_score = self._aggregate_rows(alt_rows, window)
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
    normalize: bool = False

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

    def _aggregate_rows_in_windows(self, rows: list[dict[str, Any]], windows: Sequence[ScoreWindow]) -> float:
        """Aggregate track values over one or more feature windows."""

        values: list[float] = []
        weights: list[int] = []
        for row in rows:
            weight = self._row_overlap_weight(row, windows)
            if weight <= 0:
                continue
            values.append(float(row["value"]))
            weights.append(weight)
        
        if not values:
            return float("nan")
        if self.aggregate in {"sum", "auc", "weighted_sum"}:
            return float(sum(value * weight for value, weight in zip(values, weights)))
        if self.aggregate == "mean":
            denominator = sum(weights)
            return float(sum(value * weight for value, weight in zip(values, weights)) / denominator)
        if self.aggregate == "max":
            return float(max(values))
        if self.aggregate == "min":
            return float(min(values))
        raise ValueError("max_abs_delta is computed from ref and alt rows together.")

    def score(self, prediction: PairPrediction) -> ScoringResult:
        """Return a feature-restricted track-effect score."""

        selectors = self._feature_specs()
        ref_windows = self._feature_windows(prediction.pair.ref, selectors)
        alt_windows = self._feature_windows(prediction.pair.alt, selectors)
        ref_track = prediction.ref.track(self.track, channel=self.channel)
        alt_track = prediction.alt.track(self.track, channel=self.channel)
        ref_rows = TrackWindowScorer._rows_with_values(ref_track)
        alt_rows = TrackWindowScorer._rows_with_values(alt_track)
        delta_rows = TrackWindowScorer._delta_rows(ref_rows, alt_rows, self.sign)

        if self.aggregate == "max_abs_delta":
            selected = [
                abs(float(row["delta"]))
                for row in delta_rows
                if self._row_overlap_weight(row, alt_windows) > 0
            ]
            score = max(selected) if selected else float("nan")
        else:
            ref_score = self._aggregate_rows_in_windows(ref_rows, ref_windows)
            alt_score = self._aggregate_rows_in_windows(alt_rows, alt_windows)
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

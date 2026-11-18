"""Prediction containers returned by sequence-model inference."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

from ..conditions import Condition, PooledCondition
from ..config import FULL_RETENTION, PredictionRetention, RetainedDataError
from ..sequences import AnnotatedSequence, SequencePair, _PairCoordinateMapper
from ..tracks import TrackPrediction, _aligned_delta_rows, _track_rows
from .tokenization import TokenizedSequence


def _tensor_values(value: Any) -> list[float]:
    """Convert a tensor-like value to a flat list of floats."""

    if hasattr(value, "detach"):
        value = value.detach().float().cpu().reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return [float(item) for item in value]


def _simple_value(value: Any) -> float | list[float]:
    """Convert a scalar or tensor-like value to plain Python numbers."""

    values = _tensor_values(value)
    return values[0] if len(values) == 1 else values


@dataclass
class Prediction:
    """Raw model prediction for one sequence and one condition."""

    sequence: AnnotatedSequence | None
    condition: Condition | PooledCondition
    logits: Any | None
    outputs: Mapping[str, Any]
    tokens: TokenizedSequence | None = None
    provenance: Mapping[str, Any] | None = None
    description_tokens: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def scalar(self, name: str = "expression") -> float | list[float]:
        """Return a scalar output by name."""

        if name in self.outputs:
            return _simple_value(self.outputs[name])
        if name == "expression":
            if self.logits is None:
                raise RetainedDataError(
                    "Expression logits were not retained. Keep outputs='scalars' "
                    "or enable prediction.logits in the retention policy."
                )
            return _simple_value(self.logits[:, 0:1, :].squeeze(-1))
        raise KeyError(f"Unknown scalar output {name!r}.")

    def track(self, name: str = "track", channel: int | None = 0) -> TrackPrediction:
        """Return token-level values for a model output channel."""

        if self.tokens is None:
            raise RetainedDataError(
                "Token metadata was not retained; enable prediction.tokens to build tracks."
            )
        if self.logits is None:
            raise RetainedDataError(
                "Logits were not retained; enable prediction.logits to build tracks."
            )
        input_positions = [int(row["input_position"]) for row in self.tokens.tokens]
        logits_cpu = self.logits.detach().float().cpu()
        positions = __import__("torch").tensor(input_positions, dtype=__import__("torch").long)
        if logits_cpu.ndim == 3:
            values = logits_cpu[0, positions, :]
            if values.shape[-1] == 1:
                values = values[:, 0]
            elif channel is None:
                values = values.mean(dim=-1)
            else:
                values = values[:, int(channel)]
        elif logits_cpu.ndim == 2:
            values = logits_cpu[0, positions]
        else:
            raise ValueError(f"Expected 2D or 3D logits, got shape {tuple(logits_cpu.shape)}")
        return TrackPrediction(name=name, values=[float(value) for value in values], tokens=self.tokens.tokens, channel=channel)

    def to_dict(self) -> dict[str, Any]:
        """Serialize lightweight prediction metadata."""

        return {
            "sequence": self.sequence.to_dict() if self.sequence is not None else None,
            "condition": self.condition.to_dict(),
            "outputs": {key: _simple_value(value) for key, value in self.outputs.items()},
            "provenance": dict(self.provenance or {}),
        }


@dataclass
class ExpressionPrediction(Prediction):
    """Specialized prediction where the first logit token is expression."""

    expression: float | list[float] = 0.0

    def to_frame(self):
        """Return expression and condition metadata as a one-row DataFrame."""

        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "condition": self.condition.name,
                    "expression": self.expression,
                    "sequence": self.sequence.name if self.sequence is not None else None,
                }
            ]
        )


@dataclass
class PairPrediction:
    """Reference and alternative predictions for the same variant/context."""

    ref: Prediction
    alt: Prediction
    pair: SequencePair | None
    condition: Condition | PooledCondition
    _coordinate_mapper: _PairCoordinateMapper | None = field(default=None, repr=False, compare=False)

    def delta_scalar(
        self,
        name: str = "expression",
        sign: Literal["alt-ref", "ref-alt"] = "alt-ref",
    ) -> float | list[float]:
        """Return alternative-reference scalar difference."""

        ref_values = _tensor_values(self.ref.scalar(name))
        alt_values = _tensor_values(self.alt.scalar(name))
        n = min(len(ref_values), len(alt_values))
        deltas = [alt_values[i] - ref_values[i] for i in range(n)]
        if sign == "ref-alt":
            deltas = [-value for value in deltas]
        return deltas[0] if len(deltas) == 1 else deltas

    def delta_track(
        self,
        name: str = "delta_track",
        sign: Literal["alt-ref", "ref-alt"] = "alt-ref",
        channel: int | None = 0,
        *,
        alignment: Literal["coordinate", "index"] = "coordinate",
    ) -> TrackPrediction:
        """Return a delta track in REF coordinates, with uncovered intervals as NaN.

        Index alignment reproduces legacy token-wise subtraction in ALT
        coordinates.
        """

        if alignment not in {"coordinate", "index"}:
            raise ValueError("alignment must be 'coordinate' or 'index'.")
        if sign not in {"alt-ref", "ref-alt"}:
            raise ValueError("sign must be 'alt-ref' or 'ref-alt'.")

        ref_track = self.ref.track(name="ref", channel=channel)
        alt_track = self.alt.track(name="alt", channel=channel)
        if alignment == "index":
            values = [alt - ref for ref, alt in zip(ref_track.values, alt_track.values)]
            if sign == "ref-alt":
                values = [-value for value in values]
            return TrackPrediction(
                name=name, values=values, tokens=alt_track.tokens[:len(values)], channel=channel,
            )

        mapper = self.pair.coordinate_mapper() if self.pair is not None else self._coordinate_mapper
        if mapper is None:
            raise RetainedDataError(
                "Coordinate alignment requires a sequence pair or its retained coordinate mapping."
            )
        rows = _aligned_delta_rows(_track_rows(ref_track), _track_rows(alt_track), sign, mapper=mapper)
        return TrackPrediction(
            name=name,
            values=[row["delta"] for row in rows],
            tokens=[
                {key: value for key, value in row.items() if key not in {"value", "delta"}}
                for row in rows
            ],
            channel=channel,
        )


def retain_prediction(
    prediction: Prediction,
    retention: PredictionRetention,
) -> Prediction:
    """Return ``prediction`` with fields omitted according to ``retention``."""

    if retention == FULL_RETENTION.prediction:
        return prediction
    # The logits flag controls both the dedicated field and its outputs alias.
    outputs = {
        key: (_simple_value(value) if retention.outputs == "scalars" else value)
        for key, value in prediction.outputs.items()
        if (
            key != "logits"
            or (retention.outputs == "full" and retention.logits)
        )
    }
    changes: dict[str, Any] = {
        "sequence": prediction.sequence if retention.sequence else None,
        "logits": prediction.logits if retention.logits else None,
        "outputs": outputs,
        "tokens": prediction.tokens if retention.tokens else None,
        "provenance": prediction.provenance if retention.provenance else None,
        "description_tokens": (
            prediction.description_tokens if retention.description_tokens else None
        ),
    }
    if isinstance(prediction, ExpressionPrediction):
        changes["expression"] = _simple_value(prediction.expression)
    return replace(prediction, **changes)


def retain_pair_prediction(
    prediction: PairPrediction,
    retention: PredictionRetention,
) -> PairPrediction:
    """Apply prediction retention to both alleles of a pair prediction."""

    if retention == FULL_RETENTION.prediction:
        return prediction
    return replace(
        prediction,
        ref=retain_prediction(prediction.ref, retention),
        alt=retain_prediction(prediction.alt, retention),
        pair=prediction.pair if retention.sequence else None,
        _coordinate_mapper=_retained_coordinate_mapper(
            prediction.pair, retention, existing=prediction._coordinate_mapper,
        ),
    )


def _retained_coordinate_mapper(
    pair: SequencePair | None,
    retention: PredictionRetention,
    *,
    existing: _PairCoordinateMapper | None = None,
) -> _PairCoordinateMapper | None:
    """Keep the small allele transform when only track inputs are retained."""

    if retention.logits and retention.tokens and (pair is None or not retention.sequence):
        return pair.coordinate_mapper() if pair is not None else existing
    return None

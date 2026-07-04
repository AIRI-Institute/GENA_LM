"""Prediction containers returned by :mod:`variant_api.models`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .conditions import Condition
from .sequences import AnnotatedSequence, SequencePair
from .tokenization import TokenizedSequence


def _tensor_values(value: Any) -> list[float]:
    """Convert a tensor-like value to a flat list of floats."""

    if hasattr(value, "detach"):
        value = value.detach().float().cpu().reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return [float(item) for item in value]


def _simple_value(value: Any) -> float | list[float]:
    values = _tensor_values(value)
    return values[0] if len(values) == 1 else values


@dataclass
class TrackPrediction:
    """One token-level model track with coordinate records."""

    name: str
    values: list[float]
    tokens: list[dict[str, Any]]
    channel: int | None = 0

    def to_frame(self):
        """Return track rows as a pandas DataFrame."""

        import pandas as pd

        rows = []
        for row, value in zip(self.tokens, self.values):
            rows.append({**row, self.name: value})
        return pd.DataFrame(rows)


@dataclass
class Prediction:
    """Raw model prediction for one sequence and one condition."""

    sequence: AnnotatedSequence | None
    condition: Condition
    logits: Any
    outputs: Mapping[str, Any]
    tokens: TokenizedSequence | None = None
    provenance: Mapping[str, Any] | None = None

    def scalar(self, name: str = "expression") -> float | list[float]:
        """Return a scalar output by name."""

        if name in self.outputs:
            return _simple_value(self.outputs[name])
        if name == "expression":
            return _simple_value(self.logits[:, 0:1, :].squeeze(-1))
        raise KeyError(f"Unknown scalar output {name!r}.")

    def track(self, name: str = "track", channel: int | None = 0) -> TrackPrediction:
        """Return token-level values for a model output channel."""

        if self.tokens is None:
            raise ValueError("Prediction does not contain token metadata.")
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
    pair: SequencePair
    condition: Condition

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
    ) -> TrackPrediction:
        """Return token-wise alternative-reference track difference."""

        ref_track = self.ref.track(name="ref", channel=channel)
        alt_track = self.alt.track(name="alt", channel=channel)
        n = min(len(ref_track.values), len(alt_track.values))
        values = [alt_track.values[i] - ref_track.values[i] for i in range(n)]
        if sign == "ref-alt":
            values = [-value for value in values]
        return TrackPrediction(name=name, values=values, tokens=alt_track.tokens[:n], channel=channel)

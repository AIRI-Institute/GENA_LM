"""Prediction containers returned by :mod:`variant_api.models`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
class Track:
    """One interval-level track with coordinate records."""

    name: str
    values: list[float]
    tokens: list[dict[str, Any]]
    channel: int | None = 0

    @classmethod
    def from_bw(
        cls,
        path: str | Path,
        *,
        chrom: str,
        start: int | None = None,
        end: int | None = None,
        center: int | None = None,
        size: int | None = None,
        strand: Literal["+", "-"] = "+",
        name: str | None = None,
        channel: int | None = None,
        missing_value: float = float("nan"),
    ) -> "Track":
        """Load a 1-bp-resolution track from a bigWig file.

        Coordinates follow the pyBigWig convention: 0-based, half-open.
        Pass either start/end or center/size. Minus-strand tracks are returned
        in reverse sequence order.
        """

        import math

        try:
            import pyBigWig
        except ImportError as exc:
            raise ImportError("Track.from_bw() requires pyBigWig.") from exc

        if strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'.")

        if (center is None) != (size is None):
            raise ValueError("center and size must be provided together.")

        if center is not None:
            if start is not None or end is not None:
                raise ValueError("Use either start/end or center/size, not both.")
            if int(size) <= 0:
                raise ValueError("size must be positive.")
            start = int(center) - int(size) // 2
            end = start + int(size)

        if start is None or end is None:
            raise ValueError("Provide either start/end or center/size.")

        query_start = int(start)
        query_end = int(end)
        if query_start < 0:
            raise ValueError("start must be >= 0.")
        if query_end <= query_start:
            raise ValueError("end must be greater than start.")

        bw = pyBigWig.open(str(path))
        if bw is None:
            raise OSError(f"Could not open bigWig file: {path}")

        try:
            chrom_sizes = bw.chroms()
            if chrom not in chrom_sizes:
                raise ValueError(f"Chromosome {chrom!r} is not present in {path}.")
            if query_end > int(chrom_sizes[chrom]):
                raise ValueError(
                    f"Requested interval {chrom}:{query_start}-{query_end} exceeds "
                    f"chromosome length {chrom_sizes[chrom]}."
                )

            raw_values = bw.values(chrom, query_start, query_end, numpy=False)
        finally:
            bw.close()

        values: list[float] = []
        for value in raw_values:
            if value is None:
                values.append(float(missing_value))
                continue
            value = float(value)
            values.append(float(missing_value) if math.isnan(value) else value)

        genomic_positions = list(range(query_start, query_end))
        if strand == "-":
            values = list(reversed(values))
            genomic_positions = list(reversed(genomic_positions))

        tokens = [
            {
                "input_position": i,
                "start": i,
                "end": i + 1,
                "chrom": chrom,
                "genomic_start": genomic_pos,
                "genomic_end": genomic_pos + 1,
                "strand": strand,
            }
            for i, genomic_pos in enumerate(genomic_positions)
        ]

        return cls(
            name=name or Path(path).stem,
            values=values,
            tokens=tokens,
            channel=channel,
        )

    def rescale_to_1bp(
        self,
        *,
        name: str | None = None,
        fill_value: float | None = None,
    ) -> "Track":
        """Expand interval-valued rows into one row per covered base."""

        rows = sorted(
            ({**row, "value": value} for row, value in zip(self.tokens, self.values)),
            key=lambda item: (float(item["start"]), float(item["end"])),
        )

        new_values: list[float] = []
        new_tokens: list[dict[str, Any]] = []
        last_end: int | None = None

        def append_base(pos: int, value: float, row: dict[str, Any] | None = None) -> None:
            base_row = {key: val for key, val in (row or {}).items() if key != "value"}
            base_row["start"] = pos
            base_row["end"] = pos + 1
            new_tokens.append(base_row)
            new_values.append(float(value))

        for row in rows:
            start = int(row["start"])
            end = int(row["end"])
            if end <= start:
                raise ValueError(f"Invalid track interval: start={start}, end={end}.")

            if last_end is not None:
                if start < last_end:
                    raise ValueError("Cannot rescale overlapping track intervals.")
                if fill_value is not None and start > last_end:
                    for pos in range(last_end, start):
                        append_base(pos, fill_value)

            for pos in range(start, end):
                append_base(pos, float(row["value"]), row)

            last_end = end

        return type(self)(
            name=name or self.name,
            values=new_values,
            tokens=new_tokens,
            channel=self.channel,
        )

    def to_1bp(
        self,
        *,
        name: str | None = None,
        fill_value: float | None = None,
    ) -> "Track":
        """Alias for rescale_to_1bp()."""

        return self.rescale_to_1bp(name=name, fill_value=fill_value)

    def to_frame(self):
        """Return track rows as a pandas DataFrame."""

        import pandas as pd

        rows = []
        for row, value in zip(self.tokens, self.values):
            rows.append({**row, self.name: value})
        return pd.DataFrame(rows)

    def plot(
        self,
        *,
        ax: Any | None = None,
        figsize: tuple[float, float] = (12.0, 3.0),
        title: str | None = None,
        ylabel: str | None = None,
        color: str = "tab:blue",
        color_by_sign: bool = True,
        save_path: str | Path | None = None,
    ):
        """Plot interval-level values over sequence coordinates."""

        import matplotlib.pyplot as plt

        rows = sorted(
            ({**row, "value": value} for row, value in zip(self.tokens, self.values)),
            key=lambda item: (float(item["start"]), float(item["end"])),
        )
        if not rows:
            raise ValueError(f"No rows available for track {self.name!r}.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        fig = ax.figure

        if color_by_sign:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
            for row in rows:
                value = float(row["value"])
                segment_color = "tab:green" if value > 0 else "tab:red" if value < 0 else "0.45"
                ax.hlines(
                    value,
                    float(row["start"]),
                    float(row["end"]),
                    linewidth=1.6,
                    color=segment_color,
                )
        else:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                xs.extend([float(row["start"]), float(row["end"])])
                ys.extend([float(row["value"]), float(row["value"])])
            ax.plot(xs, ys, linewidth=1.4, color=color)

        ax.set_xlabel("Sequence coordinate")
        ax.set_ylabel(ylabel or self.name)
        ax.set_title(title or self.name)
        ax.grid(axis="y", alpha=0.18)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig, ax

@dataclass
class TrackPrediction(Track):
    """Backward-compatible name for model prediction tracks."""


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

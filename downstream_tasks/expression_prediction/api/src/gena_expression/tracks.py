"""Coordinate-aware track containers, conversion, and plotting."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence

if TYPE_CHECKING:
    from .sequences import _PairCoordinateMapper


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
        """Load a 1-bp-resolution track from a bigWig file."""

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
            """Append one base-resolution value and coordinate row."""

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


class _TrackIntervalIndex:
    """Fast weighted interval queries over ordinary model-track rows.

    Model token rows are normally coordinate-sorted, non-overlapping, and
    finite. That common case uses prefix integrals. Unusual overlapping,
    unsorted, or non-finite rows retain the original scan semantics.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, value_key: str = "value") -> None:
        """Index valid track rows and prepare optional prefix integrals."""

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
        """Compute weighted area by scanning every overlapping row."""

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

    def integral_multiple(self, intervals: Iterable[tuple[int, int]]) -> tuple[float, int]:
        """Return combined weighted area and coverage for supplied intervals."""

        weighted_sum = 0.0
        total_weight = 0
        for start, end in intervals:
            window_sum, window_weight = self.integral(start, end)
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


def _track_rows(track: Track) -> list[dict[str, Any]]:
    """Combine track token metadata with its numeric values."""

    return [{**row, "value": value} for row, value in zip(track.tokens, track.values)]


def _aligned_delta_rows(
    ref_rows: list[dict[str, Any]],
    alt_rows: list[dict[str, Any]],
    sign: str,
    *,
    mapper: _PairCoordinateMapper,
    alt_index: _TrackIntervalIndex | None = None,
) -> list[dict[str, Any]]:
    """Subtract REF values from overlap-weighted ALT values in REF coordinates."""

    multiplier = -1.0 if sign == "ref-alt" else 1.0
    if alt_index is None:
        alt_index = _TrackIntervalIndex(alt_rows)
    mapped_intervals = [
        mapper.map_ref_interval_to_alt(int(row["start"]), int(row["end"]))
        for row in ref_rows
    ]
    alt_values = alt_index.weighted_means_for_sorted_intervals(mapped_intervals)
    if alt_values is None:
        alt_values = []
        for alt_start, alt_end in mapped_intervals:
            area, coverage = alt_index._scan_integral(alt_start, alt_end)
            alt_values.append(area / coverage if coverage else None)

    rows: list[dict[str, Any]] = []
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

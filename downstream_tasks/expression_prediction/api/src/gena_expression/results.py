"""Scoring result containers and export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .predictions import PairPrediction


@dataclass(frozen=True)
class ScoreWindow:
    """A sequence or token window used by a scorer."""

    start: int
    end: int
    center: int | None = None
    units: str = "sequence"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the score window."""

        return {"start": self.start, "end": self.end, "center": self.center, "units": self.units}


@dataclass
class ScoringResult:
    """A score plus the model inputs, predictions, tracks, and provenance."""

    name: str
    score: float | list[float]
    prediction: PairPrediction
    score_window: ScoreWindow | None = None
    ref_track: Any | None = None
    alt_track: Any | None = None
    delta_track: Any | None = None
    features: Any | None = None
    warnings: Sequence[str] = field(default_factory=tuple)
    provenance: Mapping[str, Any] | None = None

    @property
    def variant(self):
        """Return the variant associated with this result."""

        return self.prediction.pair.variant

    @property
    def condition(self):
        """Return the condition associated with this result."""

        return self.prediction.condition

    @property
    def input_ref(self):
        """Return the reference input sequence."""

        return self.prediction.pair.ref

    @property
    def input_alt(self):
        """Return the alternative input sequence."""

        return self.prediction.pair.alt

    @property
    def ref_prediction(self):
        """Return the reference prediction."""

        return self.prediction.ref

    @property
    def alt_prediction(self):
        """Return the alternative prediction."""

        return self.prediction.alt

    def to_frame(self):
        """Return a one-row pandas DataFrame with score metadata."""

        import pandas as pd

        variant = self.variant.to_dict() if hasattr(self.variant, "to_dict") else {}
        return pd.DataFrame(
            [
                {
                    "name": self.name,
                    "score": self.score,
                    "condition": self.condition.name,
                    "variant_id": variant.get("id"),
                    "chrom": variant.get("chrom"),
                    "pos": variant.get("pos"),
                    "ref": variant.get("ref"),
                    "alt": variant.get("alt"),
                }
            ]
        )

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize the result to JSON, optionally writing it to ``path``."""

        payload = {
            "name": self.name,
            "score": self.score,
            "score_window": self.score_window.to_dict() if self.score_window else None,
            "variant": self.variant.to_dict() if hasattr(self.variant, "to_dict") else self.variant,
            "condition": self.condition.to_dict(),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance or {}),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def plot_sequence(
        self,
        *,
        which: str = "ref",
        save_path: str | Path | None = None,
        **kwargs: Any,
    ):
        """Plot annotations on the reference or alternative input sequence."""

        sequence = self.input_ref if which == "ref" else self.input_alt
        return sequence.plot_annotations(save_path=save_path, figsize=(12, 4), **kwargs)

    def plot_variant(
        self,
        *,
        flank: int = 40,
        save_path: str | Path | None = None,
        **kwargs: Any,
    ):
        """Plot the exact sequence segment changed between ref and alt inputs."""

        return self.prediction.pair.plot_difference(flank=flank, save_path=save_path, **kwargs)

    def plot_tracks(
        self,
        *,
        figsize: tuple[float, float] = (12.0, 6.4),
        title: str | None = None,
        save_path: str | Path | None = None,
    ):
        """Plot reference, alternative, and delta ATAC tracks."""

        import matplotlib.pyplot as plt

        self._require_tracks()
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        self._plot_track_axis(axes[0], self.ref_track, "value", "ref ATAC", color="tab:blue")
        self._plot_track_axis(axes[1], self.alt_track, "value", "alt ATAC", color="tab:orange")
        self._plot_track_axis(axes[2], self.delta_track, "delta", "delta ATAC", color="tab:green", color_by_sign=True)
        for ax in axes:
            self._mark_score_window(ax)
            ax.grid(axis="y", alpha=0.18)
        axes[-1].set_xlabel("Sequence coordinate")
        fig.suptitle(title or f"{self.name}: score={self.score}")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig, axes

    def plot_delta_track(
        self,
        *,
        figsize: tuple[float, float] = (12.0, 3.0),
        title: str | None = None,
        save_path: str | Path | None = None,
    ):
        """Plot only the delta ATAC track."""

        import matplotlib.pyplot as plt

        self._require_tracks()
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_track_axis(ax, self.delta_track, "delta", "delta ATAC", color="tab:green", color_by_sign=True)
        self._mark_score_window(ax)
        ax.grid(axis="y", alpha=0.18)
        ax.set_xlabel("Sequence coordinate")
        ax.set_title(title or f"{self.name}: delta track")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig, ax

    def plot_summary(
        self,
        *,
        figsize: tuple[float, float] = (12.0, 8.0),
        include_annotations: bool = False,
        annotation_which: str = "alt",
        annotation_kwargs: Mapping[str, Any] | None = None,
        save_path: str | Path | None = None,
    ):
        """Plot variant sequence difference, optional annotations, and ATAC tracks.

        Set ``include_annotations=True`` to add one row produced by
        :meth:`AnnotatedSequence.plot_annotations`. ``annotation_which`` chooses
        the reference or alternative input sequence, and ``annotation_kwargs``
        are forwarded to ``plot_annotations``.
        """

        import matplotlib.pyplot as plt

        annotation_kwargs = dict(annotation_kwargs or {})
        annotation_sequence = self.input_ref if annotation_which == "ref" else self.input_alt

        if self.ref_track is None or self.alt_track is None or self.delta_track is None:
            if not include_annotations:
                return self.plot_variant(figsize=(figsize[0], 2.4), save_path=save_path)

            fig = plt.figure(figsize=(figsize[0], max(figsize[1] * 0.45, 4.0)))
            grid = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0])
            ax_variant = fig.add_subplot(grid[0, 0])
            self.prediction.pair.plot_difference(ax=ax_variant, title="Variant sequence difference")
            ax_annotations = fig.add_subplot(grid[1, 0])
            annotation_sequence.plot_annotations(ax=ax_annotations, **annotation_kwargs)
            fig.suptitle(f"{self.name}: score={self.score}")
            fig.tight_layout()
            if save_path is not None:
                fig.savefig(save_path, dpi=180, bbox_inches="tight")
            return fig, (ax_variant, ax_annotations)

        fig = plt.figure(figsize=figsize)
        n_rows = 5 if include_annotations else 4
        height_ratios = [1.2, 0.9, 1.0, 1.0, 1.0] if include_annotations else [1.2, 1.0, 1.0, 1.0]
        grid = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios)
        ax_variant = fig.add_subplot(grid[0, 0])
        self.prediction.pair.plot_difference(ax=ax_variant, title="Variant sequence difference")
        returned_axes = [ax_variant]
        next_row = 1
        if include_annotations:
            ax_annotations = fig.add_subplot(grid[next_row, 0])
            annotation_sequence.plot_annotations(ax=ax_annotations, **annotation_kwargs)
            returned_axes.append(ax_annotations)
            next_row += 1

        sharex = returned_axes[1] if include_annotations else None
        axes = [fig.add_subplot(grid[i, 0], sharex=sharex) for i in range(next_row, next_row + 3)]
        self._plot_track_axis(axes[0], self.ref_track, "value", "ref ATAC", color="tab:blue")
        self._plot_track_axis(axes[1], self.alt_track, "value", "alt ATAC", color="tab:orange")
        self._plot_track_axis(axes[2], self.delta_track, "delta", "delta ATAC", color="tab:green", color_by_sign=True)
        coordinate_axes = (returned_axes[1:] if include_annotations else []) + axes
        common_start, common_end = self._coordinate_extent(annotation_sequence, self.ref_track, self.alt_track, self.delta_track)
        for ax in coordinate_axes:
            ax.set_xlim(common_start, common_end)
        for ax in axes:
            self._mark_score_window(ax)
            ax.grid(axis="y", alpha=0.18)
        axes[-1].set_xlabel("Sequence coordinate")
        fig.suptitle(f"{self.name}: score={self.score}")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig, (*returned_axes, *axes)

    def save_plots(self, directory: str | Path) -> list[Path]:
        """Save standard result diagnostic plots and return written paths."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        stem = self._safe_filename(self.name)
        variant_path = directory / f"{stem}.variant.png"
        self.plot_variant(save_path=variant_path)
        paths.append(variant_path)
        if self.ref_track is not None and self.alt_track is not None and self.delta_track is not None:
            tracks_path = directory / f"{stem}.tracks.png"
            delta_path = directory / f"{stem}.delta.png"
            self.plot_tracks(save_path=tracks_path)
            self.plot_delta_track(save_path=delta_path)
            paths.extend([tracks_path, delta_path])
        return paths

    @staticmethod
    def _safe_filename(value: str) -> str:
        stem = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
        return stem.strip("._") or "result"

    def _require_tracks(self) -> None:
        if self.ref_track is None or self.alt_track is None or self.delta_track is None:
            raise ValueError("This result does not contain ref/alt/delta track rows.")

    @staticmethod
    def _records(rows: Any) -> list[dict[str, Any]]:
        if hasattr(rows, "to_dict"):
            try:
                return list(rows.to_dict(orient="records"))
            except TypeError:
                pass
        return [dict(row) for row in rows]

    @classmethod
    def _coordinate_extent(cls, sequence: Any, *tracks: Any) -> tuple[float, float]:
        """Return one x-axis extent covering sequence coordinates and track rows."""

        end = float(len(sequence))
        for rows in tracks:
            if rows is None:
                continue
            for row in cls._records(rows):
                if "end" in row:
                    end = max(end, float(row["end"]))
        return 0.0, end

    @classmethod
    def _plot_track_axis(
        cls,
        ax: Any,
        rows: Any,
        value_key: str,
        ylabel: str,
        *,
        color: str,
        color_by_sign: bool = False,
    ) -> None:
        records = cls._records(rows)
        if not records:
            raise ValueError(f"No rows available for {ylabel}.")
        sorted_records = sorted(records, key=lambda item: (float(item["start"]), float(item["end"])))
        if color_by_sign:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
            for row in sorted_records:
                if value_key not in row:
                    raise ValueError(f"Track rows do not contain {value_key!r}.")
                value = float(row[value_key])
                segment_color = "tab:green" if value > 0 else "tab:red" if value < 0 else "0.45"
                ax.hlines(
                    value,
                    float(row["start"]),
                    float(row["end"]),
                    linewidth=1.6,
                    color=segment_color,
                )
            ax.set_ylabel(ylabel)
            return

        xs: list[float] = []
        ys: list[float] = []
        for row in sorted_records:
            if value_key not in row:
                raise ValueError(f"Track rows do not contain {value_key!r}.")
            xs.extend([float(row["start"]), float(row["end"])])
            ys.extend([float(row[value_key]), float(row[value_key])])
        ax.plot(xs, ys, linewidth=1.4, color=color)
        ax.set_ylabel(ylabel)

    def _mark_score_window(self, ax: Any) -> None:
        if self.score_window is None or self.score_window.units != "sequence":
            return
        ax.axvspan(self.score_window.start, self.score_window.end, color="black", alpha=0.08)
        if self.score_window.center is not None:
            ax.axvline(self.score_window.center, color="black", linewidth=0.8, alpha=0.45)


@dataclass
class VariantReport:
    """Container for several named :class:`ScoringResult` objects."""

    results: Mapping[str, ScoringResult]

    def scores(self) -> dict[str, float | list[float]]:
        """Return ``{name: score}``."""

        return {name: result.score for name, result in self.results.items()}

    def to_frame(self):
        """Return all scores as a pandas DataFrame."""

        import pandas as pd

        return pd.concat([result.to_frame() for result in self.results.values()], ignore_index=True)

    def __getitem__(self, name: str) -> ScoringResult:
        return self.results[name]

    def plot_summary(self, *, save_path: str | Path | None = None, **kwargs: Any):
        """Plot a compact score summary for all results."""

        import matplotlib.pyplot as plt

        names = list(self.results)
        scores = [self.results[name].score for name in names]
        scalar_scores = [score[0] if isinstance(score, list) else score for score in scores]
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8.0, max(2.4, 0.35 * len(names)))))
        ax.barh(names, scalar_scores)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Score")
        ax.set_title("Variant report scores")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig, ax

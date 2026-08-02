"""Dataset-level analysis and plotting for model scoring results.

This module is standalone: it does not modify or depend on internals of the
``gena_expression`` package.  The visualizer can be built either from a list of
result objects plus a dataset, or directly from a table that already contains
one or more score columns.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


CorrelationMethod = Literal["pearson", "spearman"]
_INTERNAL_COLUMNS = ("_row_id", "_strand", "_score_name", "_score")


class ScoringResultsVisualizer:
    """Prepare scoring results once and provide fast dataset-level plots.

    Parameters
    ----------
    results
        Results in the same row order as ``dataset``. A result can be:

        - an object whose ``score`` is a scalar or ``{name: scalar}`` mapping;
        - a mapping of score names to scalar values;
        - a report-like object with a ``results`` mapping of named results.

    dataset
        A Polars or pandas DataFrame, a mapping of columns, or a sequence of
        row mappings.

    reverse_results
        Optional reverse-strand results in the same order. When supplied,
        forward, reverse, and arithmetic-average observations are retained.

    element_col, target_col
        Dataset columns containing the element/group name and experimental
        target value.

    correlation_method
        Correlation cached for every element, score, and strand.

    min_n
        Minimum number of finite paired observations required for correlation.

    Notes
    -----
    All normalization and correlation calculations happen during construction.
    Plot methods only select cached data and render it.
    """

    def __init__(
        self,
        results: Sequence[Any],
        dataset: Any,
        *,
        reverse_results: Sequence[Any] | None = None,
        element_col: str = "Element",
        target_col: str = "Value",
        correlation_method: CorrelationMethod = "pearson",
        min_n: int = 2,
    ) -> None:
        """Create a visualizer from result objects and their dataset."""
        frame = _as_polars_frame(dataset)
        _require_columns(frame, [element_col, target_col])
        _reject_reserved_columns(frame)

        forward = list(results)
        reverse = None if reverse_results is None else list(reverse_results)

        if len(forward) != frame.height:
            raise ValueError(
                "results and dataset must have the same row count: "
                f"got {len(forward)} results and {frame.height} rows."
            )
        if reverse is not None and len(reverse) != frame.height:
            raise ValueError(
                "reverse_results and dataset must have the same row count: "
                f"got {len(reverse)} results and {frame.height} rows."
            )

        forward_maps = [
            _extract_result_scores(result, row_index=i)
            for i, result in enumerate(forward)
        ]

        if reverse is None:
            observations = _observations_from_score_maps(
                frame,
                {"single": forward_maps},
            )
        else:
            reverse_maps = [
                _extract_result_scores(
                    result,
                    row_index=i,
                    label="reverse_results",
                )
                for i, result in enumerate(reverse)
            ]
            average_maps = [
                _average_score_maps(forward_map, reverse_map)
                for forward_map, reverse_map in zip(
                    forward_maps,
                    reverse_maps,
                )
            ]
            observations = _observations_from_score_maps(
                frame,
                {
                    "forward": forward_maps,
                    "reverse": reverse_maps,
                    "average": average_maps,
                },
            )

        self._setup(
            dataset=frame,
            observations=observations,
            element_col=element_col,
            target_col=target_col,
            correlation_method=correlation_method,
            min_n=min_n,
            results=forward,
        )

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        score_columns: Sequence[str],
        *,
        element_col: str = "Element",
        target_col: str = "Value",
        correlation_method: CorrelationMethod = "pearson",
        min_n: int = 2,
    ) -> "ScoringResultsVisualizer":
        """Create a visualizer from a table containing prediction columns.

        Supported file extensions are ``.csv``, ``.tsv``, ``.txt``,
        ``.parquet``, ``.ipc``, ``.feather``, ``.json``, and ``.ndjson``.
        Every name in ``score_columns`` becomes an independently selectable
        model score. All other columns remain available as plotting metadata.

        Examples
        --------
        >>> visualizer = ScoringResultsVisualizer.from_table(
        ...     "scored_variants.parquet",
        ...     score_columns=["model_a", "model_b"],
        ...     element_col="gene",
        ...     target_col="measured_effect",
        ... )
        """
        frame = _read_table(path)
        score_names = _validated_names(score_columns, "score_columns")
        _reject_reserved_columns(frame)
        _require_columns(
            frame,
            [element_col, target_col, *score_names],
        )

        metadata_columns = [
            column for column in frame.columns if column not in score_names
        ]
        observation_frames = []

        for score_name in score_names:
            observation_frames.append(
                frame.select(
                    *metadata_columns,
                    pl.int_range(0, frame.height, eager=True).alias("_row_id"),
                    pl.lit("single").alias("_strand"),
                    pl.lit(score_name).alias("_score_name"),
                    pl.col(score_name)
                    .cast(pl.Float64, strict=False)
                    .alias("_score"),
                )
            )

        observations = pl.concat(
            observation_frames,
            how="vertical_relaxed",
        )
        instance = cls.__new__(cls)
        instance._setup(
            dataset=frame,
            observations=observations,
            element_col=element_col,
            target_col=target_col,
            correlation_method=correlation_method,
            min_n=min_n,
            results=(),
        )
        return instance

    def _setup(
        self,
        *,
        dataset: pl.DataFrame,
        observations: pl.DataFrame,
        element_col: str,
        target_col: str,
        correlation_method: CorrelationMethod,
        min_n: int,
        results: Sequence[Any],
    ) -> None:
        """Validate prepared observations and cache all analysis tables."""
        if correlation_method not in {"pearson", "spearman"}:
            raise ValueError(
                "correlation_method must be 'pearson' or 'spearman'."
            )
        if min_n < 2:
            raise ValueError("min_n must be at least 2.")

        self.dataset = dataset
        self.observations = observations
        self.element_col = element_col
        self.target_col = target_col
        self.correlation_method = correlation_method
        self.min_n = int(min_n)
        self._results = tuple(results)
        self.correlations = _compute_correlations(
            observations,
            element_col=element_col,
            target_col=target_col,
            method=correlation_method,
            min_n=min_n,
        )

    @property
    def score_names(self) -> tuple[str, ...]:
        """Return score names in first-seen order."""
        return tuple(
            self.observations.get_column("_score_name")
            .unique(maintain_order=True)
            .to_list()
        )

    @property
    def elements(self) -> tuple[Any, ...]:
        """Return non-null element names in first-seen order."""
        return tuple(
            self.dataset.get_column(self.element_col)
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )

    @property
    def strands(self) -> tuple[str, ...]:
        """Return available strand/combination labels."""
        return tuple(
            self.observations.get_column("_strand")
            .unique(maintain_order=True)
            .to_list()
        )

    def prediction_frame(
        self,
        *,
        score_names: Sequence[str] | None = None,
        elements: Sequence[Any] | None = None,
        strand: str | None = None,
        long: bool = False,
    ) -> pl.DataFrame:
        """Return row-aligned predictions after optional selection.

        By default, the returned table has one row per original dataset row and
        one column per selected score. For example, five variants scored with
        ``track_window`` produce five rows and a ``track_window`` column.

        Set ``long=True`` to expose the internal analysis representation with
        one row per dataset row and score name.
        """
        selected_scores = self._select_names(
            score_names,
            available=self.score_names,
            argument="score_names",
        )
        selected_elements = self._select_names(
            elements,
            available=self.elements,
            argument="elements",
        )
        selected = self._select_observations(
            score_names=selected_scores,
            elements=selected_elements,
            strand=strand,
        )
        if long:
            return selected
        return _wide_prediction_frame(
            self.dataset,
            selected,
            score_names=selected_scores,
            element_col=self.element_col,
            elements=selected_elements,
        )

    def correlation_frame(
        self,
        *,
        score_names: Sequence[str] | None = None,
        elements: Sequence[Any] | None = None,
        strand: str | None = None,
    ) -> pl.DataFrame:
        """Return cached correlations after optional selection."""
        selected_scores = self._select_names(
            score_names,
            available=self.score_names,
            argument="score_names",
        )
        selected_elements = self._select_names(
            elements,
            available=self.elements,
            argument="elements",
        )
        selected_strand = self._resolve_strand(strand)

        return self.correlations.filter(
            pl.col("_score_name").is_in(selected_scores),
            pl.col(self.element_col).is_in(selected_elements),
            pl.col("_strand") == selected_strand,
        )

    def plot_true_vs_pred(
        self,
        element: Any,
        score_name: str,
        *,
        strand: str | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] = (5.5, 5.0),
        point_size: float = 35,
        alpha: float = 0.75,
        color: str = "#4C78A8",
        show_identity: bool = True,
        save_path: str | Path | None = None,
    ) -> tuple[Figure, Any]:
        """Plot experimental values against one prediction score."""
        selected = self._select_observations(
            score_names=[score_name],
            elements=[element],
            strand=strand,
        )
        x, y = _finite_pairs(
            selected.get_column(self.target_col),
            selected.get_column("_score"),
        )

        if len(x) == 0:
            raise ValueError(
                f"No finite observations for element={element!r}, "
                f"score_name={score_name!r}."
            )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        fig = ax.figure
        ax.scatter(y, x, s=point_size, alpha=alpha, color=color)

        if show_identity:
            lower = min(float(x.min()), float(y.min()))
            upper = max(float(x.max()), float(y.max()))
            if upper > lower:
                ax.plot(
                    [lower, upper],
                    [lower, upper],
                    linestyle="--",
                    color="black",
                    linewidth=1,
                    alpha=0.6,
                    label="y = x",
                )
                ax.legend()

        correlation = _correlation(
            x,
            y,
            method=self.correlation_method,
            min_n=self.min_n,
        )
        method_label = self.correlation_method.capitalize()
        ax.set_xlabel(str(score_name))
        ax.set_ylabel(self.target_col)
        ax.set_title(
            f"{element} — {score_name}\n"
            f"n={len(x)}, {method_label} r={correlation:.3f}"
        )
        ax.grid(alpha=0.2)
        _finish_figure(fig, save_path)
        return fig, ax

    def plot_correlation_table(
        self,
        *,
        score_names: Sequence[str] | None = None,
        elements: Sequence[Any] | None = None,
        strand: str | None = None,
        additional_rows: Mapping[str, Mapping[Any, float]] | None = None,
        summary_name: str | None = None,
        summary: Callable[[Sequence[float]], float] | None = None,
        title: str | None = None,
        value_range: tuple[float, float] = (-1.0, 1.0),
        positive_color: str = "#5F8FC8",
        negative_color: str = "#E74C3C",
        percent: bool = False,
        decimals: int = 2,
        row_height: float = 0.52,
        col_width: float = 1.25,
        label_width: float = 2.8,
        save_path: str | Path | None = None,
    ) -> tuple[Figure, Any]:
        """Plot cached correlations as a table of signed bars.

        ``additional_rows`` can add external model correlations in the form
        ``{row_name: {element: correlation}}``. Pass both ``summary_name`` and
        ``summary`` to append a final summary column.
        """
        if (summary_name is None) != (summary is None):
            raise ValueError(
                "Pass both summary_name and summary, or neither."
            )
        lo, hi = value_range
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(
                "value_range must contain two finite increasing values."
            )

        selected_scores = self._select_names(
            score_names,
            available=self.score_names,
            argument="score_names",
        )
        selected_elements = self._select_names(
            elements,
            available=self.elements,
            argument="elements",
        )
        selected_strand = self._resolve_strand(strand)
        lookup = self._correlation_lookup(selected_strand)

        rows: list[tuple[str, Mapping[Any, Any], bool]] = [
            (
                str(score_name),
                {
                    element: lookup.get((score_name, element), np.nan)
                    for element in selected_elements
                },
                True,
            )
            for score_name in selected_scores
        ]
        rows.extend(
            (str(name), dict(values), False)
            for name, values in (additional_rows or {}).items()
        )

        n_value_columns = len(selected_elements) + int(summary is not None)
        total_width = label_width + n_value_columns * col_width
        total_height = (len(rows) + 1) * row_height + 0.45
        fig, ax = plt.subplots(
            figsize=(max(8.0, total_width), max(2.8, total_height))
        )
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.axis("off")

        top = total_height - 0.12
        header_y = top - row_height
        _draw_table_cell(
            ax,
            0,
            header_y,
            label_width,
            row_height,
            "Score",
            bold=True,
        )

        for column_index, element in enumerate(selected_elements):
            _draw_table_cell(
                ax,
                label_width + column_index * col_width,
                header_y,
                col_width,
                row_height,
                str(element),
                bold=True,
                fontsize=9,
            )

        if summary is not None:
            _draw_table_cell(
                ax,
                label_width + len(selected_elements) * col_width,
                header_y,
                col_width,
                row_height,
                str(summary_name),
                bold=True,
                fontsize=9,
            )

        for row_index, (row_name, values, is_main) in enumerate(rows):
            y = header_y - (row_index + 1) * row_height
            _draw_table_cell(
                ax,
                0,
                y,
                label_width,
                row_height,
                row_name,
                bold=is_main,
                align="left",
            )

            finite_values = []
            for column_index, element in enumerate(selected_elements):
                value = _finite_float(values.get(element))
                if value is not None:
                    finite_values.append(value)
                _draw_correlation_cell(
                    ax,
                    label_width + column_index * col_width,
                    y,
                    col_width,
                    row_height,
                    value,
                    value_range=value_range,
                    positive_color=positive_color,
                    negative_color=negative_color,
                    percent=percent,
                    decimals=decimals,
                )

            if summary is not None:
                summary_value = (
                    _finite_float(summary(finite_values))
                    if finite_values
                    else None
                )
                _draw_correlation_cell(
                    ax,
                    label_width + len(selected_elements) * col_width,
                    y,
                    col_width,
                    row_height,
                    summary_value,
                    value_range=value_range,
                    positive_color=positive_color,
                    negative_color=negative_color,
                    percent=percent,
                    decimals=decimals,
                )

        ax.set_title(
            title
            or (
                f"{self.correlation_method.capitalize()} correlations "
                f"({selected_strand})"
            ),
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        _finish_figure(fig, save_path)
        return fig, ax

    def plot_correlation_heatmap(
        self,
        *,
        score_names: Sequence[str] | None = None,
        elements: Sequence[Any] | None = None,
        strand: str | None = None,
        cmap: str = "coolwarm",
        value_range: tuple[float, float] = (-1.0, 1.0),
        annotate: bool = True,
        decimals: int = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> tuple[Figure, Any]:
        """Plot a score-by-element heatmap of cached correlations."""
        selected_scores = self._select_names(
            score_names,
            available=self.score_names,
            argument="score_names",
        )
        selected_elements = self._select_names(
            elements,
            available=self.elements,
            argument="elements",
        )
        selected_strand = self._resolve_strand(strand)
        lookup = self._correlation_lookup(selected_strand)
        matrix = np.asarray(
            [
                [
                    lookup.get((score_name, element), np.nan)
                    for element in selected_elements
                ]
                for score_name in selected_scores
            ],
            dtype=float,
        )

        if figsize is None:
            figsize = (
                max(6.0, 0.75 * len(selected_elements) + 2.0),
                max(3.0, 0.55 * len(selected_scores) + 1.5),
            )
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=value_range[0],
            vmax=value_range[1],
        )
        ax.set_xticks(np.arange(len(selected_elements)))
        ax.set_xticklabels(selected_elements, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(selected_scores)))
        ax.set_yticklabels(selected_scores)

        if annotate:
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = matrix[row, column]
                    label = "–" if not np.isfinite(value) else f"{value:.{decimals}f}"
                    ax.text(
                        column,
                        row,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        ax.set_title(
            title
            or (
                f"{self.correlation_method.capitalize()} correlations "
                f"({selected_strand})"
            )
        )
        fig.colorbar(image, ax=ax, label="Correlation")
        _finish_figure(fig, save_path)
        return fig, ax

    def plot_strand_correlations(
        self,
        score_name: str,
        *,
        elements: Sequence[Any] | None = None,
        strands: Sequence[str] | None = None,
        colors: Mapping[str, str] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        save_path: str | Path | None = None,
    ) -> tuple[Figure, Any]:
        """Plot grouped correlation bars for available strands/combinations."""
        self._select_names(
            [score_name],
            available=self.score_names,
            argument="score_name",
        )
        selected_elements = self._select_names(
            elements,
            available=self.elements,
            argument="elements",
        )
        selected_strands = self._select_names(
            strands,
            available=self.strands,
            argument="strands",
        )
        palette = {
            "single": "#4C78A8",
            "forward": "#4C78A8",
            "reverse": "#F58518",
            "average": "#54A24B",
            **dict(colors or {}),
        }

        x = np.arange(len(selected_elements))
        bar_width = min(0.8 / len(selected_strands), 0.28)
        if figsize is None:
            figsize = (max(9.0, len(selected_elements) * 0.7), 5.5)
        fig, ax = plt.subplots(figsize=figsize)

        for index, selected_strand in enumerate(selected_strands):
            lookup = self._correlation_lookup(selected_strand)
            values = [
                lookup.get((score_name, element), np.nan)
                for element in selected_elements
            ]
            offset = (index - (len(selected_strands) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=selected_strand.capitalize(),
                color=palette.get(selected_strand),
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(selected_elements, rotation=45, ha="right")
        ax.set_ylabel(f"{self.correlation_method.capitalize()} correlation")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title or str(score_name))
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        _finish_figure(fig, save_path)
        return fig, ax

    def plot_variant_effects(
        self,
        element: Any,
        score_name: str,
        *,
        position_col: str,
        ref_col: str | None = None,
        alt_col: str | None = None,
        strand: str | None = None,
        relative_positions: bool = True,
        figsize: tuple[float, float] = (16, 8),
        save_path: str | Path | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Plot measured and predicted effects by position for one element.

        Alternative categories are encoded by color and reference categories by
        marker shape when ``alt_col`` and ``ref_col`` are supplied. The category
        values do not need to be nucleotide symbols, so the method is usable for
        other position-based perturbation datasets.
        """
        required = [position_col]
        if ref_col is not None:
            required.append(ref_col)
        if alt_col is not None:
            required.append(alt_col)
        _require_columns(self.observations, required)

        selected = (
            self._select_observations(
                score_names=[score_name],
                elements=[element],
                strand=strand,
            )
            .sort(
                [
                    position_col,
                    *([ref_col] if ref_col else []),
                    *([alt_col] if alt_col else []),
                ]
            )
        )
        if selected.is_empty():
            raise ValueError(
                f"No observations for element={element!r}, "
                f"score_name={score_name!r}."
            )

        positions = (
            selected.get_column(position_col)
            .cast(pl.Float64, strict=False)
            .to_numpy()
        )
        if not np.isfinite(positions).all():
            raise ValueError(
                f"{position_col!r} contains missing or non-finite values."
            )
        x = positions - positions.min() if relative_positions else positions

        references = (
            selected.get_column(ref_col).cast(pl.Utf8).to_numpy()
            if ref_col
            else np.repeat("", selected.height)
        )
        alternatives = (
            selected.get_column(alt_col).cast(pl.Utf8).to_numpy()
            if alt_col
            else np.repeat("", selected.height)
        )
        reference_values = list(dict.fromkeys(references.tolist()))
        alternative_values = list(dict.fromkeys(alternatives.tolist()))
        markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
        ref_markers = {
            value: markers[index % len(markers)]
            for index, value in enumerate(reference_values)
        }
        color_map = plt.get_cmap("tab10")
        alt_colors = {
            value: color_map(index % 10)
            for index, value in enumerate(alternative_values)
        }

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=figsize,
        )
        _draw_effect_panel(
            axes[0],
            x=x,
            values=selected.get_column(self.target_col),
            references=references,
            alternatives=alternatives,
            ref_markers=ref_markers,
            alt_colors=alt_colors,
            title=f"Measured — {self.target_col}",
        )
        _draw_effect_panel(
            axes[1],
            x=x,
            values=selected.get_column("_score"),
            references=references,
            alternatives=alternatives,
            ref_markers=ref_markers,
            alt_colors=alt_colors,
            title=f"Prediction — {score_name}",
        )

        axes[0].tick_params(axis="x", labelbottom=False)
        axes[1].set_xlabel(
            "Position relative to first observation"
            if relative_positions
            else position_col
        )

        if alt_col:
            alt_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="s",
                    markersize=8,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    label=str(value),
                )
                for value, color in alt_colors.items()
            ]
            fig.legend(
                handles=alt_handles,
                title=alt_col,
                ncol=min(5, len(alt_handles)),
                loc="lower center",
                bbox_to_anchor=(0.38, 0.01),
                frameon=False,
            )
        if ref_col:
            ref_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=marker,
                    markersize=7,
                    markerfacecolor="white",
                    markeredgecolor="#555555",
                    label=str(value),
                )
                for value, marker in ref_markers.items()
            ]
            fig.legend(
                handles=ref_handles,
                title=ref_col,
                ncol=min(5, len(ref_handles)),
                loc="lower center",
                bbox_to_anchor=(0.72, 0.01),
                frameon=False,
            )

        fig.suptitle(str(element), fontsize=16, fontweight="bold")
        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            top=0.90,
            bottom=0.17 if (ref_col or alt_col) else 0.10,
            hspace=0.10,
        )
        fig.align_ylabels(axes)
        if save_path is not None:
            _save_figure(fig, save_path)
        return fig, axes

    def plot_result(
        self,
        index: int,
        *,
        kind: Literal[
            "summary",
            "sequence",
            "variant",
            "tracks",
            "delta_track",
        ] = "summary",
        **kwargs: Any,
    ) -> Any:
        """Delegate an individual diagnostic plot to the original result."""
        if not self._results:
            raise ValueError(
                "Individual result objects are unavailable because this "
                "visualizer was created with from_table()."
            )
        result = self._results[index]
        method_name = {
            "summary": "plot_summary",
            "sequence": "plot_sequence",
            "variant": "plot_variant",
            "tracks": "plot_tracks",
            "delta_track": "plot_delta_track",
        }[kind]
        method = getattr(result, method_name, None)
        if method is None:
            raise TypeError(
                f"Result {index} does not provide {method_name}()."
            )
        return method(**kwargs)

    def _select_observations(
        self,
        *,
        score_names: Sequence[str] | None,
        elements: Sequence[Any] | None,
        strand: str | None,
    ) -> pl.DataFrame:
        """Select normalized observations with validated public arguments."""
        selected_scores = self._select_names(
            score_names,
            available=self.score_names,
            argument="score_names",
        )
        selected_elements = self._select_names(
            elements,
            available=self.elements,
            argument="elements",
        )
        selected_strand = self._resolve_strand(strand)
        return self.observations.filter(
            pl.col("_score_name").is_in(selected_scores),
            pl.col(self.element_col).is_in(selected_elements),
            pl.col("_strand") == selected_strand,
        )

    def _resolve_strand(self, strand: str | None) -> str:
        """Choose the most useful default strand and validate it."""
        if strand is None:
            strand = "average" if "average" in self.strands else self.strands[0]
        if strand not in self.strands:
            raise KeyError(
                f"Unknown strand {strand!r}; available: {self.strands!r}."
            )
        return strand

    def _correlation_lookup(
        self,
        strand: str,
    ) -> dict[tuple[str, Any], float]:
        """Return cached correlations keyed by score and element."""
        selected = self.correlations.filter(pl.col("_strand") == strand)
        return {
            (row["_score_name"], row[self.element_col]): row["correlation"]
            for row in selected.iter_rows(named=True)
        }

    @staticmethod
    def _select_names(
        requested: Sequence[Any] | None,
        *,
        available: Sequence[Any],
        argument: str,
    ) -> list[Any]:
        """Validate an optional ordered subset against available values."""
        selected = list(available if requested is None else requested)
        if not selected:
            raise ValueError(f"{argument} must not be empty.")
        missing = [value for value in selected if value not in available]
        if missing:
            raise KeyError(
                f"Unknown {argument}: {missing!r}; available: "
                f"{list(available)!r}."
            )
        return list(dict.fromkeys(selected))


def _read_table(path: str | Path) -> pl.DataFrame:
    """Read a supported table file into a Polars DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pl.read_csv(path, separator="\t")
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".ipc", ".feather"}:
        return pl.read_ipc(path)
    if suffix == ".ndjson":
        return pl.read_ndjson(path)
    if suffix == ".json":
        return pl.read_json(path)
    raise ValueError(
        f"Unsupported table extension {suffix!r}. Supported extensions: "
        ".csv, .tsv, .txt, .parquet, .ipc, .feather, .json, .ndjson."
    )


def _as_polars_frame(dataset: Any) -> pl.DataFrame:
    """Convert a supported in-memory dataset to a Polars DataFrame."""
    if isinstance(dataset, pl.DataFrame):
        return dataset.clone()
    if hasattr(dataset, "to_dict") and hasattr(dataset, "columns"):
        # pandas offers a stable records representation across versions.
        return pl.DataFrame(dataset.to_dict(orient="records"))
    if isinstance(dataset, Mapping):
        return pl.DataFrame(dict(dataset))
    return pl.DataFrame([dict(row) for row in dataset])


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing dataframe columns: {missing!r}.")


def _reject_reserved_columns(frame: pl.DataFrame) -> None:
    """Reject dataset columns reserved for the internal long representation."""
    conflicts = [
        column for column in _INTERNAL_COLUMNS if column in frame.columns
    ]
    if conflicts:
        raise ValueError(
            "Dataset uses reserved visualization columns: "
            f"{conflicts!r}. Rename them before creating the visualizer."
        )


def _validated_names(names: Sequence[str], argument: str) -> list[str]:
    """Return unique non-empty string names while preserving order."""
    values = [str(name) for name in names]
    if not values:
        raise ValueError(f"{argument} must contain at least one name.")
    if any(not name for name in values):
        raise ValueError(f"{argument} must not contain empty names.")
    if len(set(values)) != len(values):
        raise ValueError(f"{argument} must not contain duplicates.")
    return values


def _extract_result_scores(
    result: Any,
    *,
    row_index: int,
    label: str = "results",
) -> dict[str, float]:
    """Flatten one scalar, feature result, or report into named scores."""
    report_results = getattr(result, "results", None)
    if isinstance(report_results, Mapping):
        flattened: dict[str, float] = {}
        for report_name, child in report_results.items():
            child_scores = _score_value_to_mapping(
                getattr(child, "score", child),
                default_name=str(report_name),
                context=f"{label}[{row_index}].results[{report_name!r}]",
            )
            for child_name, value in child_scores.items():
                key = (
                    str(report_name)
                    if child_name == str(report_name)
                    else f"{report_name}:{child_name}"
                )
                if key in flattened:
                    raise ValueError(
                        f"Duplicate flattened score name {key!r} at "
                        f"{label}[{row_index}]."
                    )
                flattened[key] = value
        return flattened

    score = getattr(result, "score", result)
    default_name = _stable_result_name(result)
    return _score_value_to_mapping(
        score,
        default_name=default_name,
        context=f"{label}[{row_index}]",
    )


def _stable_result_name(result: Any) -> str:
    """Remove a batch item prefix from a result name when one is present.

    ``VariantInterpreter`` names batched results as
    ``"<variant label>:<scorer name>"``. The variant label is row identity, not
    a distinct score type, so only the final scorer-name component belongs in
    the visualization schema.
    """
    name = str(getattr(result, "name", "score") or "score")

    variant_id = None
    identity = getattr(result, "identity", None)
    if identity is not None:
        variant_id = getattr(identity, "variant_id", None) or getattr(
            identity,
            "id",
            None,
        )

    if variant_id is None:
        try:
            variant = result.variant
        except Exception:
            variant = None
        variant_id = getattr(variant, "id", None)

    if variant_id is not None:
        prefix = f"{variant_id}:"
        if name.startswith(prefix):
            return name[len(prefix) :]

    # This fallback covers compact genomic results that retained a name such as
    # ``chr11:5250012:G>C:track_window`` but not full variant identity.
    if name.count(":") >= 3:
        return name.rsplit(":", 1)[-1]
    return name


def _score_value_to_mapping(
    value: Any,
    *,
    default_name: str,
    context: str,
) -> dict[str, float]:
    """Convert one supported score value into a flat numeric mapping."""
    if isinstance(value, Mapping):
        return {
            str(name): _score_float(score, context=f"{context}[{name!r}]")
            for name, score in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return {
            f"{default_name}[{index}]": _score_float(
                score,
                context=f"{context}[{index}]",
            )
            for index, score in enumerate(value)
        }
    return {
        default_name: _score_float(value, context=context),
    }


def _score_float(value: Any, *, context: str) -> float:
    """Convert a scalar score to float, representing missing values as NaN."""
    if value is None:
        return float("nan")
    if hasattr(value, "score"):
        value = value.score
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{context} must contain a scalar numeric score, got {value!r}."
        ) from exc


def _average_score_maps(
    forward: Mapping[str, float],
    reverse: Mapping[str, float],
) -> dict[str, float]:
    """Average common finite scores and retain missing pairs as NaN."""
    names = list(dict.fromkeys([*forward, *reverse]))
    averaged = {}
    for name in names:
        forward_value = forward.get(name, np.nan)
        reverse_value = reverse.get(name, np.nan)
        averaged[name] = (
            float((forward_value + reverse_value) / 2.0)
            if np.isfinite(forward_value) and np.isfinite(reverse_value)
            else float("nan")
        )
    return averaged


def _observations_from_score_maps(
    dataset: pl.DataFrame,
    maps_by_strand: Mapping[str, Sequence[Mapping[str, float]]],
) -> pl.DataFrame:
    """Build the canonical long observation table from row-wise score maps."""
    rows = dataset.to_dicts()
    records = []

    for strand, score_maps in maps_by_strand.items():
        for row_id, (row, score_map) in enumerate(zip(rows, score_maps)):
            # Emit only scores belonging to this result. Expanding the union of
            # all row-specific names would create a false Cartesian product.
            for score_name, score_value in score_map.items():
                records.append(
                    {
                        **row,
                        "_row_id": row_id,
                        "_strand": strand,
                        "_score_name": score_name,
                        "_score": score_value,
                    }
                )

    if not records:
        raise ValueError("No scores were found in the supplied results.")
    return pl.DataFrame(records)


def _wide_prediction_frame(
    dataset: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    score_names: Sequence[str],
    element_col: str,
    elements: Sequence[Any],
) -> pl.DataFrame:
    """Return one dataset row with one column for every selected score."""
    base = (
        dataset.with_row_index("_row_id")
        .filter(pl.col(element_col).is_in(elements))
    )

    # from_table() datasets already contain the score columns. Drop them here
    # so the selected strand values can be joined back under their clean names.
    existing_scores = [
        score_name for score_name in score_names if score_name in base.columns
    ]
    if existing_scores:
        base = base.drop(existing_scores)

    for score_name in score_names:
        score_frame = (
            observations.filter(pl.col("_score_name") == score_name)
            .select(
                "_row_id",
                pl.col("_score").alias(score_name),
            )
        )
        if score_frame.get_column("_row_id").n_unique() != score_frame.height:
            raise ValueError(
                f"Score {score_name!r} occurs more than once for a dataset "
                "row in the selected strand."
            )
        base = base.join(score_frame, on="_row_id", how="left")

    return base.drop("_row_id")


def _compute_correlations(
    observations: pl.DataFrame,
    *,
    element_col: str,
    target_col: str,
    method: CorrelationMethod,
    min_n: int,
) -> pl.DataFrame:
    """Compute one cached correlation per strand, score, and element."""
    records = []
    groups = observations.partition_by(
        ["_strand", "_score_name", element_col],
        maintain_order=True,
    )

    for group in groups:
        x, y = _finite_pairs(
            group.get_column(target_col),
            group.get_column("_score"),
        )
        records.append(
            {
                "_strand": group["_strand"][0],
                "_score_name": group["_score_name"][0],
                element_col: group[element_col][0],
                "n": len(x),
                "correlation": _correlation(
                    x,
                    y,
                    method=method,
                    min_n=min_n,
                ),
            }
        )

    return pl.DataFrame(records)


def _finite_pairs(
    left: pl.Series,
    right: pl.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite float arrays from two Polars series."""
    x = left.cast(pl.Float64, strict=False).to_numpy()
    y = right.cast(pl.Float64, strict=False).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: CorrelationMethod,
    min_n: int,
) -> float:
    """Calculate a guarded Pearson or Spearman correlation."""
    if (
        len(x) < min_n
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return float("nan")
    if method == "spearman":
        x = _average_ranks(x)
        y = _average_ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including tied values."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0

    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None for an unavailable value."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _draw_table_cell(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str = "",
    *,
    bold: bool = False,
    align: Literal["left", "center"] = "center",
    fontsize: int = 10,
) -> None:
    """Draw one bordered text cell on a Matplotlib axis."""
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.8,
        )
    )
    text_x = x + 0.06 if align == "left" else x + width / 2
    ax.text(
        text_x,
        y + height / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
    )


def _draw_correlation_cell(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    value: float | None,
    *,
    value_range: tuple[float, float],
    positive_color: str,
    negative_color: str,
    percent: bool,
    decimals: int,
) -> None:
    """Draw one signed correlation bar and its numeric label."""
    _draw_table_cell(ax, x, y, width, height)
    if value is None:
        ax.text(x + width / 2, y + height / 2, "–", ha="center", va="center")
        return

    max_abs = max(abs(value_range[0]), abs(value_range[1]))
    zero_x = x + width / 2
    bar_width = min(abs(value) / max_abs, 1.0) * (width / 2 - 0.06)
    bar_height = height * 0.72
    bar_y = y + (height - bar_height) / 2
    bar_x = zero_x if value >= 0 else zero_x - bar_width
    color = positive_color if value >= 0 else negative_color

    ax.add_patch(
        Rectangle(
            (bar_x, bar_y),
            bar_width,
            bar_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.9,
        )
    )
    ax.plot(
        [zero_x, zero_x],
        [y, y + height],
        color="#202020",
        linewidth=0.35,
    )
    label = (
        f"{value * 100:.{decimals}f}%"
        if percent
        else f"{value:.{decimals}f}"
    )
    ax.text(
        x + width - 0.05,
        y + height / 2,
        label,
        ha="right",
        va="center",
        fontsize=9,
    )


def _draw_effect_panel(
    ax: Any,
    *,
    x: np.ndarray,
    values: pl.Series,
    references: np.ndarray,
    alternatives: np.ndarray,
    ref_markers: Mapping[str, str],
    alt_colors: Mapping[str, Any],
    title: str,
) -> None:
    """Draw one measured-or-predicted positional effect panel."""
    numeric_values = values.cast(pl.Float64, strict=False).to_numpy()
    valid = np.isfinite(numeric_values)
    if not valid.any():
        raise ValueError(f"{title!r} has no finite values to plot.")

    point_colors = [alt_colors[value] for value in alternatives[valid]]
    ax.vlines(
        x[valid],
        ymin=0,
        ymax=numeric_values[valid],
        colors=point_colors,
        linewidth=0.8,
        alpha=0.75,
        zorder=1,
    )
    for reference, marker in ref_markers.items():
        mask = valid & (references == reference)
        if mask.any():
            ax.scatter(
                x[mask],
                numeric_values[mask],
                c=[alt_colors[value] for value in alternatives[mask]],
                marker=marker,
                s=22,
                linewidths=0,
                zorder=2,
            )

    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("Effect")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    lower = min(0.0, float(numeric_values[valid].min()))
    upper = max(0.0, float(numeric_values[valid].max()))
    if lower == upper:
        ax.set_ylim(-0.5, 0.5)
    else:
        padding = 0.08 * (upper - lower)
        ax.set_ylim(lower - padding, upper + padding)


def _finish_figure(
    fig: Figure,
    save_path: str | Path | None,
) -> None:
    """Apply tight layout and optionally save a figure."""
    fig.tight_layout()
    if save_path is not None:
        _save_figure(fig, save_path)


def _save_figure(fig: Figure, path: str | Path) -> None:
    """Create the destination directory and save a figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def load_correlation_json(path: str | Path) -> pl.DataFrame:
    """Load legacy nested correlation JSON into a long Polars table.

    Expected input shape:
    ``{element: {strand_or_score_name: correlation}}``.
    This helper is intentionally separate from plotting so callers can inspect,
    transform, or combine the data before visualization.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records = []
    for element, values in payload.items():
        if not isinstance(values, Mapping):
            raise TypeError(
                f"Expected a mapping for element {element!r}, "
                f"got {type(values).__name__}."
            )
        for name, value in values.items():
            records.append(
                {
                    "element": element,
                    "name": str(name),
                    "correlation": _finite_float(value),
                }
            )
    return pl.DataFrame(records)

"""Plotting helpers backing :class:`gena_expression.ism.ISM`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .core import ISM


_ALT_COLORS = {
    "A": "#159447",
    "C": "#2468A2",
    "G": "#F2A51A",
    "T": "#DC2638",
    "-": "#6B7280",
}
_REF_MARKERS = {
    "A": "s",
    "C": "o",
    "G": "^",
    "T": "D",
}
_DNA_ALPHABET = "ACGT"


def _validated_frame(
    ism: "ISM",
    scorer_name: str,
    mutation_type: Literal["substitution", "deletion", "all"],
):
    import polars as pl

    if scorer_name not in ism.scorer_names:
        raise KeyError(f"Unknown scorer {scorer_name!r}; available: {ism.scorer_names!r}.")
    if mutation_type not in {"substitution", "deletion", "all"}:
        raise ValueError("mutation_type must be 'substitution', 'deletion', or 'all'.")
    frame = ism.variants
    if mutation_type != "all":
        frame = frame.filter(pl.col("mutation_type") == mutation_type)
    if frame.is_empty():
        raise ValueError(f"No {mutation_type!r} variants are available.")
    return frame


def plot_variant_effects(
    ism: "ISM",
    scorer_name: str,
    *,
    mutation_type: Literal["substitution", "deletion", "all"] = "substitution",
    target_col: str | None = None,
    position_mode: Literal["region", "sequence"] = "region",
    figsize: tuple[float, float] | None = None,
):
    """Plot optional measured and predicted effects for one ISM library."""

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from matplotlib.lines import Line2D

    frame = _validated_frame(ism, scorer_name, mutation_type)
    if target_col is not None and target_col not in frame.columns:
        raise KeyError(f"Target column {target_col!r} is not present in the ISM dataframe.")
    if position_mode not in {"region", "sequence"}:
        raise ValueError("position_mode must be 'region' or 'sequence'.")

    position_col = "position_in_region" if position_mode == "region" else "position"
    value_columns = ([target_col] if target_col is not None else []) + [scorer_name]
    labels = ([f"Measured — {target_col}"] if target_col is not None else []) + [
        f"Model prediction — {scorer_name}"
    ]
    frame = frame.sort([position_col, "ref", "alt"])
    positions = frame.get_column(position_col).cast(pl.Float64).to_numpy()
    references = frame.get_column("ref").cast(pl.Utf8).str.to_uppercase().to_numpy()
    alternatives = (
        frame.get_column("alt")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .replace("", "-")
        .to_numpy()
    )
    invalid_ref = sorted(set(references) - set(_REF_MARKERS))
    invalid_alt = sorted(set(alternatives) - set(_ALT_COLORS))
    if invalid_ref:
        raise ValueError(f"Unsupported reference alleles: {invalid_ref}")
    if invalid_alt:
        raise ValueError(f"Unsupported alternative alleles: {invalid_alt}")

    nrows = len(value_columns)
    if figsize is None:
        figsize = (16.0, 4.5 if nrows == 1 else 8.0)
    fig, axes_value = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=figsize)
    axes = np.atleast_1d(axes_value)

    for ax, value_col, title in zip(axes, value_columns, labels):
        values = frame.get_column(value_col).cast(pl.Float64).to_numpy()
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError(f"{value_col!r} has no finite values to plot.")
        point_colors = [_ALT_COLORS[base] for base in alternatives[valid]]
        ax.vlines(
            positions[valid],
            ymin=0,
            ymax=values[valid],
            colors=point_colors,
            linewidth=0.8,
            alpha=0.75,
            zorder=1,
        )
        for reference, marker in _REF_MARKERS.items():
            mask = valid & (references == reference)
            if mask.any():
                ax.scatter(
                    positions[mask],
                    values[mask],
                    c=[_ALT_COLORS[base] for base in alternatives[mask]],
                    marker=marker,
                    s=22,
                    linewidths=0,
                    zorder=2,
                )
        ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("Variant effect")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        lower = min(0.0, float(values[valid].min()))
        upper = max(0.0, float(values[valid].max()))
        if lower == upper:
            ax.set_ylim(-0.5, 0.5)
        else:
            padding = 0.08 * (upper - lower)
            ax.set_ylim(lower - padding, upper + padding)

    axes[-1].set_xlabel(
        "Position relative to region start (bp)"
        if position_mode == "region"
        else "Sequence position (0-based)"
    )
    axes[-1].ticklabel_format(axis="x", style="plain", useOffset=False)
    alt_bases = [base for base in _ALT_COLORS if base in set(alternatives)]
    alt_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=8,
            markerfacecolor=_ALT_COLORS[base],
            markeredgecolor=_ALT_COLORS[base],
            label=base,
        )
        for base in alt_bases
    ]
    ref_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="#555555",
            label=base,
        )
        for base, marker in _REF_MARKERS.items()
        if base in set(references)
    ]
    fig.legend(
        handles=alt_handles,
        title="Alternative",
        ncol=max(1, len(alt_handles)),
        loc="lower center",
        bbox_to_anchor=(0.40, 0.01),
        frameon=False,
    )
    fig.legend(
        handles=ref_handles,
        title="Reference",
        ncol=max(1, len(ref_handles)),
        loc="lower center",
        bbox_to_anchor=(0.72, 0.01),
        frameon=False,
    )
    title = ism.sequence.name or ism.region.name or "ISM"
    fig.suptitle(str(title), fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.22, hspace=0.10)
    fig.align_ylabels(axes)
    return fig, axes


def plot_substitution_matrix(
    ism: "ISM",
    scorer_name: str,
    *,
    cmap: str = "coolwarm",
    center: float | None = 0.0,
    figsize: tuple[float, float] = (16.0, 4.0),
):
    """Plot a four-row alternative-base saturation mutagenesis heatmap."""

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    frame = _validated_frame(ism, scorer_name, "substitution")
    width = ism.region.end - ism.region.start
    bases = list(_DNA_ALPHABET)
    matrix = np.full((len(bases), width), np.nan, dtype=float)
    for row in frame.select("position_in_region", "alt", scorer_name).to_dicts():
        value = row[scorer_name]
        if value is not None:
            matrix[bases.index(row["alt"]), int(row["position_in_region"])] = float(value)

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError(f"{scorer_name!r} has no finite substitution scores to plot.")
    kwargs = {}
    if center is not None:
        limit = max(abs(float(finite.min()) - center), abs(float(finite.max()) - center))
        kwargs = {"vmin": center - limit, "vmax": center + limit}
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        **kwargs,
    )
    ax.set_yticks(range(len(bases)), labels=bases)
    ax.set_ylabel("Alternative base")
    ax.set_xlabel("Position relative to region start (bp)")
    ax.set_title(f"{ism.sequence.name or ism.region.name or 'ISM'} — {scorer_name}")
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Variant effect")
    fig.tight_layout()
    return fig, ax

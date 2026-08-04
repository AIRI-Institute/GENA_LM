#!/usr/bin/env python3
"""Plot saturation-mutagenesis HDF5 scores with alternative colors and reference markers."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DNA = "ACGT"
ALT_COLORS = {"A": "#159447", "C": "#23679f", "G": "#f39c12", "T": "#e5243b"}
REF_MARKERS = {"A": "s", "C": "o", "G": "^", "T": "D"}
ORIENTATION_INDEX = {"forward": 0, "reverse_complement": 1}


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def load_ref_alt(
    path: Path,
    *,
    tss_id: str | None,
    orientation: str,
    relative_start: int | None,
    relative_end: int | None,
) -> tuple[str, np.ndarray]:
    """Return rows of (TSS-relative position, ref, alt, delta) from one HDF5 TSS."""

    with h5py.File(path, "r") as handle:
        ids = [_text(value) for value in handle["tss/tss_id"][:]]
        if tss_id is None:
            if len(ids) != 1:
                raise ValueError("HDF5 contains multiple TSSs; select one with --tss-id")
            index = 0
        else:
            matches = [i for i, value in enumerate(ids) if value == tss_id]
            if len(matches) != 1:
                raise ValueError(f"Expected one TSS named {tss_id!r}, found {len(matches)}")
            index = matches[0]
        if int(handle["tss/status"][index]) != 1:
            raise ValueError(f"TSS {ids[index]!r} is not complete")

        flat_start, flat_end = map(int, handle["tss/offsets"][index : index + 2])
        genomic_start_0based = int(handle["tss/genomic_start_0based"][index])
        tss_1based = int(handle["tss/tss_position_1based"][index])
        positions_1based = genomic_start_0based + handle["position_offset"][flat_start:flat_end] + 1
        relative_positions = positions_1based - tss_1based
        ref_codes = handle["ref_base"][flat_start:flat_end]
        deltas = handle["scores"][flat_start:flat_end, :, ORIENTATION_INDEX[orientation]]

        rows: list[tuple[int, str, str, float]] = []
        for relative_position, ref_code, values in zip(relative_positions, ref_codes, deltas):
            position = int(relative_position)
            if relative_start is not None and position < relative_start:
                continue
            if relative_end is not None and position > relative_end:
                continue
            ref = DNA[int(ref_code)]
            for alt, delta in zip((base for base in DNA if base != ref), values):
                rows.append((position, ref, alt, float(delta)))

    dtype = [("position", "i8"), ("ref", "U1"), ("alt", "U1"), ("delta", "f4")]
    return ids[index], np.asarray(rows, dtype=dtype)


def plot_ref_alt(
    rows: np.ndarray,
    *,
    title: str,
    output: Path,
    dpi: int,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Draw notebook-style stems, using alt color and ref marker shape."""

    figure, axis = plt.subplots(figsize=(18, 5))
    axis.axhline(0, color="0.25", linewidth=0.8)
    for alt in DNA:
        for ref in DNA:
            selected = rows[(rows["alt"] == alt) & (rows["ref"] == ref)]
            if not len(selected):
                continue
            color = ALT_COLORS[alt]
            axis.vlines(selected["position"], 0, selected["delta"], color=color, alpha=0.45, linewidth=0.7)
            axis.scatter(
                selected["position"], selected["delta"], color=color,
                marker=REF_MARKERS[ref], s=18, linewidths=0, zorder=3,
            )

    alt_handles = [
        plt.Line2D([], [], color=ALT_COLORS[base], marker="s", linestyle="None", label=base)
        for base in DNA
    ]
    ref_handles = [
        plt.Line2D([], [], marker=REF_MARKERS[base], markerfacecolor="none", markeredgecolor="0.3", linestyle="None", label=base)
        for base in DNA
    ]
    legend_alt = axis.legend(handles=alt_handles, title="Alternative", ncol=4, loc="upper center", bbox_to_anchor=(0.38, -0.18), frameon=False)
    axis.add_artist(legend_alt)
    axis.legend(handles=ref_handles, title="Reference", ncol=4, loc="upper center", bbox_to_anchor=(0.68, -0.18), frameon=False)
    axis.set(title=title, xlabel="Position relative to TSS (bp)", ylabel="Variant effect (alt − ref ATAC sum)")
    if y_min is not None or y_max is not None:
        current_min, current_max = axis.get_ylim()
        axis.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max,
        )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Completed shard or merged HDF5 file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tss-id")
    parser.add_argument("--orientation", choices=tuple(ORIENTATION_INDEX), default="forward")
    parser.add_argument("--relative-start", type=int)
    parser.add_argument("--relative-end", type=int)
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--y-min", type=float, help="Clip the displayed y-axis minimum")
    parser.add_argument("--y-max", type=float, help="Clip the displayed y-axis maximum")
    args = parser.parse_args()
    tss_id, rows = load_ref_alt(
        args.input,
        tss_id=args.tss_id,
        orientation=args.orientation,
        relative_start=args.relative_start,
        relative_end=args.relative_end,
    )
    if not len(rows):
        raise ValueError("No variants remain after applying the requested interval")
    title = args.title or f"{tss_id} — {args.orientation}"
    if args.y_min is not None and args.y_max is not None and args.y_min >= args.y_max:
        raise ValueError("--y-min must be smaller than --y-max")
    plot_ref_alt(
        rows,
        title=title,
        output=args.output,
        dpi=args.dpi,
        y_min=args.y_min,
        y_max=args.y_max,
    )
    print(f"wrote {len(rows)} variants to {args.output}")


if __name__ == "__main__":
    main()

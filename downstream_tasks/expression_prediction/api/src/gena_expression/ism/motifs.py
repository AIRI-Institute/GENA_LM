"""Scan a reference region for MEME motifs under an independent-base null."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..sequences import CoordinateMap, Feature


_DNA = "ACGT"
_BASE_INDEX = {base: index for index, base in enumerate(_DNA)}
_BACKGROUND_FRACTION = 1e-4
_SCORE_SCALE = 100
_HIT_SCHEMA = {
    "motif_id": pl.String,
    "tf": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "start_in_region": pl.Int64,
    "end_in_region": pl.Int64,
    "strand": pl.String,
    "source_chrom": pl.String,
    "source_start": pl.Int64,
    "source_end": pl.Int64,
    "source_strand": pl.String,
    "score": pl.Float64,
    "p_value": pl.Float64,
    "p_value_threshold": pl.Float64,
    "overlaps_peak": pl.Boolean,
}


@dataclass(frozen=True)
class _Motif:
    identifier: str
    tf: str
    probabilities: np.ndarray


@dataclass(frozen=True)
class _GenomicInterval:
    chrom: str
    start: int
    end: int
    strand: str


def genomic_interval(coordinate_map: CoordinateMap, start: int, end: int) -> _GenomicInterval | None:
    """Return a projection only for contiguous coverage in one chromosome and orientation."""
    segments = sorted(coordinate_map.slice(start, end).segments, key=lambda segment: segment.seq_start)
    if not segments or segments[-1].seq_end != end - start:
        return None

    first = segments[0]
    if first.chrom is None or first.strand not in {"+", "-"}:
        return None
    reverse = first.strand == "-"
    expected_boundary = first.source_end if reverse else first.source_start
    next_start = 0
    for segment in segments:
        if (
            segment.seq_start != next_start
            or segment.source != first.source
            or segment.chrom != first.chrom
            or segment.strand != first.strand
            or segment.source_start is None
            or segment.source_end is None
            or segment.source_start < 0
            or segment.source_end - segment.source_start != segment.seq_end - segment.seq_start
        ):
            return None
        boundary = segment.source_end if reverse else segment.source_start
        if boundary != expected_boundary:
            return None
        expected_boundary = segment.source_start if reverse else segment.source_end
        next_start = segment.seq_end
    if reverse:
        return _GenomicInterval(first.chrom, segments[-1].source_start, first.source_end, first.strand)
    return _GenomicInterval(first.chrom, first.source_start, segments[-1].source_end, first.strand)


def _read_meme(path: str | Path) -> tuple[np.ndarray, list[_Motif]]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines or not lines[0].startswith("MEME version "):
        raise ValueError(f"{path!s} does not start with a MEME version header.")

    motif_indices = [
        index for index, line in enumerate(lines) if line.split(maxsplit=1)[0] == "MOTIF"
    ]
    if not motif_indices:
        raise ValueError(f"No motifs found in {path!s}.")

    header = lines[:motif_indices[0]]
    alphabet = next((line for line in header if line.startswith("ALPHABET")), "ALPHABET= ACGT")
    if "=" not in alphabet or "".join(alphabet.split("=", 1)[1].split()) != _DNA:
        raise ValueError("Expected the MEME DNA alphabet ACGT.")

    for line in header:
        if line.startswith("strands:"):
            declared = line.partition(":")[2].split()
            if not declared or set(declared) - {"+", "-"} or len(set(declared)) != len(declared):
                raise ValueError("MEME strands must be declared using '+' and/or '-'.")

    background = np.full(4, 0.25)
    for index, line in enumerate(header):
        if line.startswith("Background letter frequencies"):
            entries = " ".join(header[index + 1:]).split()
            if len(entries) != 8 or set(entries[::2]) != set(_DNA):
                raise ValueError("Expected four MEME background frequencies for A, C, G, and T.")
            frequencies = dict(zip(entries[::2], entries[1::2]))
            background = np.array([float(frequencies[base]) for base in _DNA])
            if not (np.isfinite(background).all() and (background > 0).all() and np.isclose(background.sum(), 1)):
                raise ValueError("MEME background frequencies must be positive and sum to one.")
            background /= background.sum()
            break

    motifs = []
    identifiers = set()
    for start, end in zip(motif_indices, motif_indices[1:] + [len(lines)]):
        motif_header = lines[start].split(maxsplit=2)
        if len(motif_header) < 2:
            raise ValueError("Expected a motif ID after MOTIF.")
        _, identifier, *alternate = motif_header
        if identifier in identifiers:
            raise ValueError(f"Duplicate MEME motif ID {identifier!r}.")
        identifiers.add(identifier)
        for index in range(start + 1, end):
            if lines[index].startswith("letter-probability matrix:"):
                break
        else:
            raise ValueError(f"Missing letter-probability matrix for {identifier!r}.")
        fields = dict(re.findall(r"(\w+)\s*=\s*(\S+)", lines[index]))
        width = int(fields.get("w", "0"))
        if width <= 0 or fields.get("alength") != "4" or index + width >= end:
            raise ValueError(f"Invalid DNA matrix dimensions for {identifier!r}.")
        try:
            probabilities = np.array(
                [[float(value) for value in row.split()] for row in lines[index + 1:index + width + 1]]
            )
        except ValueError as exc:
            raise ValueError(f"Malformed MEME probability matrix for {identifier!r}.") from exc
        if (
            probabilities.shape != (width, 4)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=0.002)
        ):
            raise ValueError(f"Invalid MEME probability matrix for {identifier!r}.")
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        tf = alternate[0] if alternate else identifier.split(".H14CORE.", 1)[0]
        motifs.append(_Motif(identifier, tf, probabilities))
    return background, motifs


def _score_tail(scores: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Compute the inclusive tail for sums of nonnegative integer position scores."""
    mass = np.ones(1)
    for position in scores:
        updated = np.zeros(len(mass) + int(position.max()))
        for shift, probability in zip(position, background):
            updated[shift:shift + len(mass)] += probability * mass
        mass = updated
    return np.clip(mass[::-1].cumsum()[::-1], 0.0, 1.0)


def scan_motifs(
    sequence: str,
    coordinate_map: CoordinateMap,
    region: Feature,
    motif_file: str | Path,
    *,
    p_value_threshold: float,
    peak_intervals: list[tuple[int, int]] | None,
) -> pl.DataFrame:
    """Scan reference windows and return one row per motif, start, and strand."""
    background, motifs = _read_meme(motif_file)
    bases = np.fromiter(
        (_BASE_INDEX.get(base, -1) for base in sequence[region.start:region.end]), dtype=np.int8
    )
    motifs_by_width: dict[int, list[_Motif]] = defaultdict(list)
    for motif in motifs:
        motifs_by_width[motif.probabilities.shape[0]].append(motif)

    rows = []
    for width, same_width in motifs_by_width.items():
        if width > len(bases):
            continue
        windows = np.lib.stride_tricks.sliding_window_view(bases, width)
        valid = np.flatnonzero((windows >= 0).all(axis=1))
        if not valid.size:
            continue
        valid_windows = windows[valid]
        positions = np.arange(width)
        for motif in same_width:
            # Score the reverse-complemented motif against the same
            # forward-coordinate background on the minus strand.
            for strand, probabilities in (
                ("+", motif.probabilities),
                ("-", motif.probabilities[::-1, ::-1]),
            ):
                smoothed = (1 - _BACKGROUND_FRACTION) * probabilities + _BACKGROUND_FRACTION * background
                log_odds = np.log2(smoothed / background)
                quantized = np.rint(log_odds * _SCORE_SCALE).astype(int)
                quantized -= quantized.min(axis=1, keepdims=True)
                tail = _score_tail(quantized, background)
                p_values = tail[quantized[positions, valid_windows].sum(axis=1)]
                for hit in np.flatnonzero(p_values <= p_value_threshold):
                    local_start = int(valid[hit])
                    start = region.start + local_start
                    end = start + width
                    projected = genomic_interval(coordinate_map, start, end)
                    overlaps_peak = None
                    if peak_intervals is not None:
                        overlaps_peak = any(
                            start < peak_end and peak_start < end for peak_start, peak_end in peak_intervals
                        )
                    source_strand = None
                    if projected is not None:
                        source_strand = "+" if strand == projected.strand else "-"
                    rows.append({
                        "motif_id": motif.identifier,
                        "tf": motif.tf,
                        "start": start,
                        "end": end,
                        "start_in_region": local_start,
                        "end_in_region": local_start + width,
                        "strand": strand,
                        "source_chrom": projected.chrom if projected else None,
                        "source_start": projected.start if projected else None,
                        "source_end": projected.end if projected else None,
                        "source_strand": source_strand,
                        "score": float(log_odds[positions, valid_windows[hit]].sum()),
                        "p_value": float(p_values[hit]),
                        "p_value_threshold": p_value_threshold,
                        "overlaps_peak": overlaps_peak,
                    })
    return pl.DataFrame(rows, schema=_HIT_SCHEMA).sort(["start", "end", "tf", "motif_id", "strand"])

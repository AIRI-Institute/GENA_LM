"""Build SNV effect profiles and compare them with positioned TF motif hits."""

from __future__ import annotations

from bisect import bisect_right
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from .motifs import _motif_log_odds, limit_motif_overlaps

if TYPE_CHECKING:
    from .core import ISM


_BASE_INDEX = {base: index for index, base in enumerate("ACGT")}


def substitution_effect_matrix(variants: pl.DataFrame, scorer_name: str, width: int) -> np.ndarray:
    """Place available ALT effects in a position-by-base matrix."""
    matrix = np.full((width, 4), np.nan)
    for position, alt, value in variants.select("position_in_region", "alt", scorer_name).iter_rows():
        if value is not None:
            matrix[position, _BASE_INDEX[alt]] = float(value)
    return matrix


def attribution_peaks(
    attribution: np.ndarray,
    threshold: float | None,
    *,
    mode: Literal["quantile", "absolute"],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Find runs of highlighted bases in a region-relative attribution track."""
    if mode not in ("quantile", "absolute"):
        raise ValueError("attribution_threshold_mode must be 'quantile' or 'absolute'.")
    if threshold is None and mode == "quantile":
        threshold = 0.90
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or threshold < 0
        or (mode == "quantile" and threshold > 1)
    ):
        if mode == "quantile":
            raise ValueError("attribution_threshold must be a quantile between 0 and 1.")
        raise ValueError("Absolute attribution_threshold must be a finite nonnegative number.")

    magnitudes = np.abs(attribution)
    finite = np.isfinite(magnitudes)
    cutoff = threshold
    if mode == "quantile" and finite.any():
        cutoff = np.quantile(magnitudes[finite], threshold)
    highlighted = finite & (magnitudes > 0) & (magnitudes >= cutoff)
    edges = np.flatnonzero(np.diff(np.r_[False, highlighted, False])).tolist()
    return highlighted, list(zip(edges[::2], edges[1::2]))


def _weighted_cosine(observed: np.ndarray, motif: np.ndarray, weights: np.ndarray) -> float | None:
    # Rescale the ISM effects before taking a norm so their units do not affect similarity.
    scale = np.abs(observed).max()
    if scale == 0:
        return None
    root_weights = np.sqrt(weights / weights.max())[:, None]
    observed = (observed / scale) * root_weights
    motif = motif * root_weights
    denominator = np.linalg.norm(observed) * np.linalg.norm(motif)
    if denominator == 0:
        return None
    return float(np.clip(np.sum(observed * motif) / denominator, -1, 1))


def rank_motif_hits(
    ism: ISM,
    scorer_name: str,
    *,
    attribution_threshold: float | None,
    attribution_threshold_mode: Literal["quantile", "absolute"],
    min_profile_similarity: float,
    min_peak_coverage: float,
    profile_direction: Literal["positive", "negative", "either"],
    max_overlapping_tfs: int | None,
) -> pl.DataFrame:
    """Rank overlapping candidates from the reference scan for each attribution peak."""
    for name, value in (
        ("min_profile_similarity", min_profile_similarity),
        ("min_peak_coverage", min_peak_coverage),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if profile_direction not in ("positive", "negative", "either"):
        raise ValueError("profile_direction must be 'positive', 'negative', or 'either'.")
    if ism.tf_hits is None:
        raise ValueError("Call tf_annotate() before ranking motif hits.")

    hits = ism.tf_hits
    positions = ism.position_attributions(scorer_name)
    attribution = positions.get_column("attribution").to_numpy()
    _, peaks = attribution_peaks(attribution, attribution_threshold, mode=attribution_threshold_mode)
    polarity = {"positive": 1, "negative": -1, "either": 0}[profile_direction]
    rows = []
    if peaks and not hits.is_empty():
        background = getattr(ism, "_tf_background", None)
        if background is None:
            raise ValueError("Call tf_annotate() again to load the motif background for ranking.")

        substitutions = ism.variants.filter(pl.col("mutation_type") == "substitution")
        effects = substitution_effect_matrix(substitutions, scorer_name, len(attribution))
        for index, ref in enumerate(positions.get_column("ref")):
            if np.isfinite(attribution[index]):
                effects[index, _BASE_INDEX[ref]] = 0.0
        effects -= (0.25 * effects).sum(axis=1, keepdims=True)

        magnitudes = np.abs(attribution)
        peak_ends = [end for _, end in peaks]
        peak_weights = [magnitudes[start:end] / magnitudes[start:end].max() for start, end in peaks]
        peak_totals = [weights.sum() for weights in peak_weights]
        pwms = {}
        for hit in hits.iter_rows(named=True):
            start, end = hit["start_in_region"], hit["end_in_region"]
            key = hit["motif_id"], hit["strand"]
            for peak_index in range(bisect_right(peak_ends, start), len(peaks)):
                peak_start, peak_end = peaks[peak_index]
                if peak_start >= end:
                    break
                left, right = max(start, peak_start), min(end, peak_end)
                covered = peak_weights[peak_index][left - peak_start:right - peak_start].sum()
                coverage = float(covered / peak_totals[peak_index])
                if coverage < min_peak_coverage:
                    continue
                if key not in pwms:
                    probabilities = ism._tf_matrices[key[0]]
                    if key[1] == "-":
                        probabilities = probabilities[::-1, ::-1]
                    pwm = _motif_log_odds(probabilities, background)
                    pwms[key] = pwm - pwm.mean(axis=1, keepdims=True)
                similarity = _weighted_cosine(
                    effects[left:right],
                    pwms[key][left - start:right - start],
                    magnitudes[left:right],
                )
                if similarity is None:
                    continue
                agreement = polarity * similarity if polarity else abs(similarity)
                if agreement > 0 and agreement >= min_profile_similarity:
                    rows.append({
                        **hit,
                        "peak_start": peak_start + ism.region.start,
                        "peak_end": peak_end + ism.region.start,
                        "peak_coverage": coverage,
                        "profile_similarity": similarity,
                        "_agreement": agreement,
                    })

    ranked = (
        pl.DataFrame(rows, schema={
            **hits.schema,
            "peak_start": pl.Int64,
            "peak_end": pl.Int64,
            "peak_coverage": pl.Float64,
            "profile_similarity": pl.Float64,
            "_agreement": pl.Float64,
        })
        .sort(
            ["peak_start", "_agreement", "p_value", "start", "end", "motif_id", "strand"],
            descending=[False, True, False, False, False, False, False],
        )
        .with_columns(pl.col("peak_start").cum_count().over("peak_start").alias("peak_rank"))
        .sort(
            ["peak_rank", "_agreement", "p_value", "peak_start", "start", "end", "motif_id", "strand"],
            descending=[False, True, False, False, False, False, False, False],
        )
        .drop("_agreement")
    )
    if max_overlapping_tfs is None:
        return ranked

    hit_columns = ["start_in_region", "end_in_region", "motif_id", "tf", "strand"]
    unique_hits = ranked.select(hit_columns).unique(maintain_order=True)
    accepted = limit_motif_overlaps(
        unique_hits.iter_rows(), origin=0, width=len(attribution),
        max_overlapping_tfs=max_overlapping_tfs,
    )
    selected = pl.DataFrame(list(accepted), schema=unique_hits.schema, orient="row")
    return ranked.join(selected, on=hit_columns, how="semi", maintain_order="left")

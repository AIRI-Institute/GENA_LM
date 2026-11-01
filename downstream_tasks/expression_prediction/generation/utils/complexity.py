"""Insertion-sequence population complexity metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from statistics import fmean, pstdev

import polars as pl


DNA_ALPHABET = frozenset("ACGTN")


def _normalize_sequences(sequences: Iterable[str]) -> tuple[str, ...]:
    """Return uppercase, equal-length DNA strings after strict validation."""

    normalized: list[str] = []
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise TypeError(f"Sequence at index {index} must be a string.")
        value = sequence.upper()
        if not value:
            raise ValueError(f"Sequence at index {index} must not be empty.")
        invalid = sorted(set(value) - DNA_ALPHABET)
        if invalid:
            raise ValueError(
                f"Sequence at index {index} contains unsupported symbols: "
                + ", ".join(invalid)
            )
        normalized.append(value)

    if not normalized:
        raise ValueError("Population must contain at least one sequence.")

    sequence_length = len(normalized[0])
    for index, sequence in enumerate(normalized[1:], start=1):
        if len(sequence) != sequence_length:
            raise ValueError(
                f"Sequence at index {index} has length {len(sequence)}; "
                f"expected {sequence_length}."
            )
    return tuple(normalized)


def compute_population_complexity(
    sequences: Iterable[str],
) -> dict[str, int | float]:
    """Compute exact insertion-population diversity and composition metrics.

    Hamming diversity is the mean mismatch fraction over every distinct pair,
    calculated from per-position symbol counts in ``O(population_size *
    sequence_length)`` time. Position entropy includes ``N`` as a fifth symbol.
    GC fractions use the full sequence length as their denominator, while the
    ambiguous-base fraction reports the contribution from ``N`` explicitly.
    Standard deviation is the population standard deviation (``ddof=0``).
    """

    population = _normalize_sequences(sequences)
    population_size = len(population)
    sequence_length = len(population[0])

    mismatch_sum = 0.0
    entropy_sum = 0.0
    polymorphic_position_count = 0
    pair_denominator = population_size * (population_size - 1)

    for position in zip(*population):
        counts = Counter(position)
        if len(counts) > 1:
            polymorphic_position_count += 1
        if pair_denominator:
            match_fraction = sum(
                count * (count - 1) for count in counts.values()
            ) / pair_denominator
            mismatch_sum += 1.0 - match_fraction
        entropy_sum += -sum(
            (count / population_size) * math.log2(count / population_size)
            for count in counts.values()
        )

    gc_fractions = [
        (sequence.count("G") + sequence.count("C")) / sequence_length
        for sequence in population
    ]
    ambiguous_base_count = sum(sequence.count("N") for sequence in population)
    unique_sequence_count = len(set(population))

    return {
        "population_size": population_size,
        "sequence_length": sequence_length,
        "unique_sequence_count": unique_sequence_count,
        "unique_sequence_fraction": unique_sequence_count / population_size,
        "mean_pairwise_hamming_fraction": mismatch_sum / sequence_length,
        "mean_position_entropy_bits": entropy_sum / sequence_length,
        "polymorphic_position_count": polymorphic_position_count,
        "polymorphic_position_fraction": (
            polymorphic_position_count / sequence_length
        ),
        "gc_fraction_mean": fmean(gc_fractions),
        "gc_fraction_std": pstdev(gc_fractions),
        "ambiguous_base_fraction": (
            ambiguous_base_count / (population_size * sequence_length)
        ),
    }


def compute_frame_population_complexity(
    population: pl.DataFrame,
    *,
    sequence_column: str = "region_sequence",
) -> dict[str, int | float]:
    """Compute complexity from one Polars population table."""

    if sequence_column not in population.columns:
        raise KeyError(
            f"Population table has no sequence column {sequence_column!r}."
        )
    if population.schema[sequence_column] != pl.String:
        raise TypeError(f"Population column {sequence_column!r} must be String.")
    if population[sequence_column].null_count():
        raise ValueError(
            f"Population column {sequence_column!r} must not contain nulls."
        )
    return compute_population_complexity(population[sequence_column].to_list())

"""Reusable crossover-and-mutation offspring generators."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal


DNA_ALPHABET = "ACGT"


def _validate_generator_parameters(
    *,
    crossover_probability: float,
    mutations_per_child: int,
    random_seed: int,
    parent_selection: str,
) -> None:
    """Validate parameters shared by serial and parallel generators."""

    if isinstance(crossover_probability, bool) or not isinstance(
        crossover_probability,
        Real,
    ):
        raise TypeError("crossover_probability must be a real number.")
    if not math.isfinite(float(crossover_probability)) or not (
        0.0 <= crossover_probability <= 1.0
    ):
        raise ValueError("crossover_probability must be between 0 and 1.")
    if isinstance(mutations_per_child, bool) or not isinstance(
        mutations_per_child,
        int,
    ):
        raise TypeError("mutations_per_child must be an integer.")
    if mutations_per_child < 0:
        raise ValueError("mutations_per_child must be non-negative.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer.")
    if not isinstance(parent_selection, str):
        raise TypeError("parent_selection must be a string.")
    if parent_selection not in {"uniform", "shifted_fitness"}:
        raise ValueError(
            "parent_selection must be 'uniform' or 'shifted_fitness'."
        )


def _normalize_parents(
    parent_records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Extract immutable candidate-ID/sequence pairs from parent records."""

    parents: list[tuple[str, str]] = []
    for record in parent_records:
        if "candidate_id" not in record:
            raise KeyError("Parent records must contain 'candidate_id'.")
        if "region_sequence" not in record:
            raise KeyError("Parent records must contain 'region_sequence'.")
        candidate_id = record["candidate_id"]
        region_sequence = record["region_sequence"]
        if not isinstance(candidate_id, str):
            raise TypeError("Parent candidate_id values must be strings.")
        if not isinstance(region_sequence, str):
            raise TypeError("Parent region_sequence values must be strings.")
        parents.append((candidate_id, region_sequence))
    return tuple(parents)


def _parent_weights(
    parent_records: Sequence[Mapping[str, Any]],
    *,
    parent_selection: str,
) -> tuple[float, ...] | None:
    """Return shifted fitness weights, or ``None`` for uniform selection."""

    if parent_selection == "uniform":
        return None

    fitness_values: list[float] = []
    for record in parent_records:
        if "fitness" not in record:
            raise KeyError(
                "Parent records must contain 'fitness' when "
                "parent_selection='shifted_fitness'."
            )
        fitness = record["fitness"]
        if isinstance(fitness, bool) or not isinstance(fitness, Real):
            raise TypeError("Parent fitness values must be real numbers.")
        value = float(fitness)
        if not math.isfinite(value):
            raise ValueError("Parent fitness values must be finite.")
        fitness_values.append(value)

    if not fitness_values:
        return None

    minimum = min(fitness_values)
    weights = tuple(value - minimum for value in fitness_values)
    if not all(math.isfinite(weight) for weight in weights):
        raise ValueError("Parent fitness values span too large a range.")
    maximum = max(weights)
    if maximum == 0.0:
        return None
    return tuple(weight / maximum for weight in weights)


def _context_integer(context: Mapping[str, Any], key: str) -> int:
    """Read one non-negative integer from an optimizer generator context."""

    if key not in context:
        raise KeyError(f"Generator context must contain {key!r}.")
    value = context[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Generator context {key!r} must be an integer.")
    if value < 0:
        raise ValueError(f"Generator context {key!r} must be non-negative.")
    return value


def _child_seed(base_seed: int, generation: int, child_index: int) -> int:
    """Derive a stable seed independent of process scheduling and worker count."""

    payload = f"{base_seed}:{generation}:{child_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _select_parent(
    rng: random.Random,
    parents: tuple[tuple[str, str], ...],
    parent_weights: tuple[float, ...] | None,
) -> tuple[str, str]:
    """Select one parent uniformly or by its configured fitness weight."""

    if parent_weights is None:
        return rng.choice(parents)
    return rng.choices(parents, weights=parent_weights, k=1)[0]


def _select_parent_pair(
    rng: random.Random,
    parents: tuple[tuple[str, str], ...],
    parent_weights: tuple[float, ...] | None,
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Select two distinct parents uniformly or by fitness weight."""

    if parent_weights is None:
        parent_a, parent_b = rng.sample(parents, 2)
        return parent_a, parent_b

    indexes = tuple(range(len(parents)))
    first_index = rng.choices(indexes, weights=parent_weights, k=1)[0]
    remaining_indexes = tuple(index for index in indexes if index != first_index)
    remaining_weights = tuple(parent_weights[index] for index in remaining_indexes)
    if any(remaining_weights):
        second_index = rng.choices(
            remaining_indexes,
            weights=remaining_weights,
            k=1,
        )[0]
    else:
        second_index = rng.choice(remaining_indexes)
    return parents[first_index], parents[second_index]


def _generate_child(
    parents: tuple[tuple[str, str], ...],
    *,
    parent_weights: tuple[float, ...] | None,
    seed: int,
    crossover_probability: float,
    mutations_per_child: int,
) -> dict[str, Any]:
    """Generate one child with the notebook's crossover/mutation procedure."""

    rng = random.Random(seed)
    parent_a_id, parent_a_sequence = _select_parent(
        rng,
        parents,
        parent_weights,
    )
    child = parent_a_sequence
    parent_ids = [parent_a_id]

    if (
        len(parents) >= 2
        and len(parent_a_sequence) >= 2
        and rng.random() < crossover_probability
    ):
        (parent_a_id, parent_a_sequence), (
            parent_b_id,
            parent_b_sequence,
        ) = _select_parent_pair(rng, parents, parent_weights)
        crossover_position = rng.randrange(1, len(parent_a_sequence))
        child = (
            parent_a_sequence[:crossover_position]
            + parent_b_sequence[crossover_position:]
        )
        parent_ids = [parent_a_id, parent_b_id]

    child_bases = list(child)
    n_mutations = min(mutations_per_child, len(child_bases))
    mutation_positions = rng.sample(range(len(child_bases)), n_mutations)
    for position in mutation_positions:
        current_base = child_bases[position]
        alternatives = DNA_ALPHABET.replace(current_base, "")
        child_bases[position] = rng.choice(alternatives or DNA_ALPHABET)

    return {
        "region_sequence": "".join(child_bases),
        "parent_ids": parent_ids,
    }


@dataclass(frozen=True)
class _BatchJob:
    """Pickle-friendly description of one contiguous offspring batch."""

    parents: tuple[tuple[str, str], ...]
    parent_weights: tuple[float, ...] | None
    start: int
    count: int
    generation: int
    random_seed: int
    crossover_probability: float
    mutations_per_child: int


def _generate_batch(job: _BatchJob) -> list[dict[str, Any]]:
    """Generate one ordered batch inside a worker process."""

    return [
        _generate_child(
            job.parents,
            parent_weights=job.parent_weights,
            seed=_child_seed(
                job.random_seed,
                job.generation,
                child_index,
            ),
            crossover_probability=job.crossover_probability,
            mutations_per_child=job.mutations_per_child,
        )
        for child_index in range(job.start, job.start + job.count)
    ]


def _batch_ranges(total: int, n_batches: int) -> list[tuple[int, int]]:
    """Return contiguous ``(start, count)`` ranges covering ``total`` items."""

    quotient, remainder = divmod(total, n_batches)
    ranges: list[tuple[int, int]] = []
    start = 0
    for batch_index in range(n_batches):
        count = quotient + int(batch_index < remainder)
        if count:
            ranges.append((start, count))
            start += count
    return ranges


@dataclass(frozen=True)
class SequentialCrossoverMutationGenerator:
    """Generate offspring serially with deterministic per-child seeds.

    Use this callable with ``offspring_generator_mode="records"``. It implements
    the notebook's parent selection, optional one-point crossover, and distinct
    point mutations while also retaining parent candidate IDs. Set
    ``parent_selection="shifted_fitness"`` to sample parents proportionally to
    their fitness after shifting the lowest fitness to zero.
    """

    crossover_probability: float = 0.6
    mutations_per_child: int = 20
    random_seed: int = 0
    parent_selection: Literal["uniform", "shifted_fitness"] = "uniform"

    def __post_init__(self) -> None:
        """Validate configuration once at construction."""

        _validate_generator_parameters(
            crossover_probability=self.crossover_probability,
            mutations_per_child=self.mutations_per_child,
            random_seed=self.random_seed,
            parent_selection=self.parent_selection,
        )

    def __call__(
        self,
        parent_records: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> Iterable[dict[str, Any]]:
        """Yield exactly ``context['n_offspring']`` ordered children."""

        parents = _normalize_parents(parent_records)
        if not parents:
            return
        parent_weights = _parent_weights(
            parent_records,
            parent_selection=self.parent_selection,
        )
        generation = _context_integer(context, "generation")
        n_offspring = _context_integer(context, "n_offspring")
        for child_index in range(n_offspring):
            yield _generate_child(
                parents,
                parent_weights=parent_weights,
                seed=_child_seed(
                    self.random_seed,
                    generation,
                    child_index,
                ),
                crossover_probability=self.crossover_probability,
                mutations_per_child=self.mutations_per_child,
            )


@dataclass
class ParallelCrossoverMutationGenerator:
    """Generate deterministic offspring batches in spawned worker processes.

    The generator owns a lazily created :class:`ProcessPoolExecutor`. Use it as
    a context manager, or call :meth:`close` after evolution. Spawned workers
    never receive the model or other CUDA objects. Serial and parallel
    generators return identical children for equal parameters and seeds.
    Parent selection can be uniform or shifted-fitness weighted.
    """

    max_workers: int | None = None
    batches_per_worker: int = 1
    crossover_probability: float = 0.6
    mutations_per_child: int = 20
    random_seed: int = 0
    parent_selection: Literal["uniform", "shifted_fitness"] = "uniform"
    _executor: ProcessPoolExecutor | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate process and mutation configuration."""

        _validate_generator_parameters(
            crossover_probability=self.crossover_probability,
            mutations_per_child=self.mutations_per_child,
            random_seed=self.random_seed,
            parent_selection=self.parent_selection,
        )
        for name, value in (
            ("max_workers", self.max_workers),
            ("batches_per_worker", self.batches_per_worker),
        ):
            if value is None and name == "max_workers":
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

    @property
    def worker_count(self) -> int:
        """Return the configured or platform-default worker count."""

        return self.max_workers or (os.cpu_count() or 1)

    def _pool(self) -> ProcessPoolExecutor:
        """Create the persistent spawn-based pool on first use."""

        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._executor

    def __call__(
        self,
        parent_records: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> Iterable[dict[str, Any]]:
        """Yield process-generated children in deterministic child-index order."""

        parents = _normalize_parents(parent_records)
        if not parents:
            return
        parent_weights = _parent_weights(
            parent_records,
            parent_selection=self.parent_selection,
        )
        generation = _context_integer(context, "generation")
        n_offspring = _context_integer(context, "n_offspring")
        if n_offspring == 0:
            return

        n_batches = min(
            n_offspring,
            self.worker_count * self.batches_per_worker,
        )
        jobs = [
            _BatchJob(
                parents=parents,
                parent_weights=parent_weights,
                start=start,
                count=count,
                generation=generation,
                random_seed=self.random_seed,
                crossover_probability=self.crossover_probability,
                mutations_per_child=self.mutations_per_child,
            )
            for start, count in _batch_ranges(n_offspring, n_batches)
        ]
        for batch in self._pool().map(_generate_batch, jobs):
            yield from batch

    def close(self) -> None:
        """Wait for pending work, then close the owned process pool."""

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "ParallelCrossoverMutationGenerator":
        """Return this generator; the process pool remains lazily initialized."""

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the process pool when leaving a context-manager block."""

        self.close()


__all__ = [
    "ParallelCrossoverMutationGenerator",
    "SequentialCrossoverMutationGenerator",
]

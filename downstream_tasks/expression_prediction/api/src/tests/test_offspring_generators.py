"""Tests for reusable serial and process-based offspring generators."""

from __future__ import annotations

import unittest
from collections import Counter

import gena_expression as gx


class OffspringGeneratorTests(unittest.TestCase):
    """Verify deterministic crossover/mutation output across backends."""

    @staticmethod
    def _parents():
        return [
            {
                "candidate_id": "parent-a",
                "region_sequence": "AAAAAA",
            },
            {
                "candidate_id": "parent-b",
                "region_sequence": "CCCCCC",
            },
            {
                "candidate_id": "parent-c",
                "region_sequence": "GGGGGG",
            },
        ]

    @staticmethod
    def _context():
        return {
            "generation": 3,
            "n_offspring": 12,
        }

    @classmethod
    def _weighted_parents(cls):
        parents = cls._parents()
        for parent, fitness in zip(parents, (-2.0, 1.0, 4.0), strict=True):
            parent["fitness"] = fitness
        return parents

    def test_sequential_generator_is_repeatable_and_records_parents(self):
        generator = gx.SequentialCrossoverMutationGenerator(
            crossover_probability=0.6,
            mutations_per_child=2,
            random_seed=17,
        )

        first = list(generator(self._parents(), self._context()))
        second = list(generator(self._parents(), self._context()))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertTrue(all(len(row["region_sequence"]) == 6 for row in first))
        self.assertTrue(all(row["parent_ids"] for row in first))

    def test_parallel_generator_matches_sequential_for_any_batching(self):
        parameters = {
            "crossover_probability": 0.6,
            "mutations_per_child": 2,
            "random_seed": 17,
        }
        expected = list(
            gx.SequentialCrossoverMutationGenerator(**parameters)(
                self._parents(),
                self._context(),
            )
        )

        with gx.ParallelCrossoverMutationGenerator(
            max_workers=2,
            batches_per_worker=2,
            **parameters,
        ) as generator:
            observed = list(generator(self._parents(), self._context()))

        self.assertEqual(observed, expected)

    def test_shifted_fitness_supports_negative_values(self):
        parents = self._weighted_parents()
        parents[1]["fitness"] = -2.0
        generator = gx.SequentialCrossoverMutationGenerator(
            crossover_probability=0.0,
            mutations_per_child=0,
            random_seed=17,
            parent_selection="shifted_fitness",
        )

        children = list(generator(parents, self._context()))

        self.assertTrue(
            all(child["parent_ids"] == ["parent-c"] for child in children)
        )

    def test_shifted_fitness_controls_selection_probability(self):
        generator = gx.SequentialCrossoverMutationGenerator(
            crossover_probability=0.0,
            mutations_per_child=0,
            random_seed=17,
            parent_selection="shifted_fitness",
        )
        context = {"generation": 3, "n_offspring": 600}

        children = list(generator(self._weighted_parents(), context))
        counts = Counter(child["parent_ids"][0] for child in children)

        self.assertEqual(counts["parent-a"], 0)
        self.assertGreater(counts["parent-b"], 0)
        self.assertGreater(counts["parent-c"], counts["parent-b"])

    def test_equal_fitness_falls_back_to_uniform_selection(self):
        parents = self._weighted_parents()
        for parent in parents:
            parent["fitness"] = 5.0
        parameters = {
            "crossover_probability": 0.6,
            "mutations_per_child": 2,
            "random_seed": 17,
        }

        uniform = list(
            gx.SequentialCrossoverMutationGenerator(**parameters)(
                parents,
                self._context(),
            )
        )
        weighted = list(
            gx.SequentialCrossoverMutationGenerator(
                parent_selection="shifted_fitness",
                **parameters,
            )(
                parents,
                self._context(),
            )
        )

        self.assertEqual(weighted, uniform)

    def test_weighted_crossover_uses_distinct_parents(self):
        parents = self._weighted_parents()
        parents[1]["fitness"] = -2.0
        generator = gx.SequentialCrossoverMutationGenerator(
            crossover_probability=1.0,
            mutations_per_child=0,
            random_seed=17,
            parent_selection="shifted_fitness",
        )

        children = list(generator(parents, self._context()))

        self.assertTrue(all(len(child["parent_ids"]) == 2 for child in children))
        self.assertTrue(
            all(len(set(child["parent_ids"])) == 2 for child in children)
        )

    def test_shifted_fitness_requires_finite_real_fitness(self):
        generator = gx.SequentialCrossoverMutationGenerator(
            parent_selection="shifted_fitness"
        )
        cases = (
            ({}, KeyError),
            ({"fitness": True}, TypeError),
            ({"fitness": float("nan")}, ValueError),
            ({"fitness": float("inf")}, ValueError),
        )

        for update, error in cases:
            parents = self._parents()
            for parent in parents:
                parent.update(update)
            with self.subTest(update=update):
                with self.assertRaises(error):
                    list(generator(parents, self._context()))

    def test_parent_selection_mode_is_validated(self):
        with self.assertRaises(ValueError):
            gx.SequentialCrossoverMutationGenerator(
                parent_selection="fitness",
            )

    def test_parallel_generator_matches_weighted_sequential(self):
        parameters = {
            "crossover_probability": 0.6,
            "mutations_per_child": 2,
            "random_seed": 17,
            "parent_selection": "shifted_fitness",
        }
        expected = list(
            gx.SequentialCrossoverMutationGenerator(**parameters)(
                self._weighted_parents(),
                self._context(),
            )
        )

        with gx.ParallelCrossoverMutationGenerator(
            max_workers=2,
            batches_per_worker=2,
            **parameters,
        ) as generator:
            observed = list(generator(self._weighted_parents(), self._context()))

        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()

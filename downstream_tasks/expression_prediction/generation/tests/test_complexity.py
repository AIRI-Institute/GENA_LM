"""Population complexity metric tests."""

from __future__ import annotations

import unittest

import polars as pl

from utils.complexity import (
    compute_frame_population_complexity,
    compute_population_complexity,
)


class PopulationComplexityTests(unittest.TestCase):
    """Verify exact diversity, composition, and input-validation behavior."""

    def test_one_substitution_has_exact_hamming_entropy_and_polymorphism(self):
        metrics = compute_population_complexity(["AAAA", "AAAT"])

        self.assertEqual(metrics["population_size"], 2)
        self.assertEqual(metrics["sequence_length"], 4)
        self.assertEqual(metrics["unique_sequence_count"], 2)
        self.assertEqual(metrics["unique_sequence_fraction"], 1.0)
        self.assertEqual(metrics["mean_pairwise_hamming_fraction"], 0.25)
        self.assertEqual(metrics["mean_position_entropy_bits"], 0.25)
        self.assertEqual(metrics["polymorphic_position_count"], 1)
        self.assertEqual(metrics["polymorphic_position_fraction"], 0.25)
        self.assertEqual(metrics["gc_fraction_mean"], 0.0)
        self.assertEqual(metrics["gc_fraction_std"], 0.0)
        self.assertEqual(metrics["ambiguous_base_fraction"], 0.0)

    def test_duplicates_and_lowercase_are_normalized(self):
        metrics = compute_population_complexity(["acgt", "ACGT"])

        self.assertEqual(metrics["unique_sequence_count"], 1)
        self.assertEqual(metrics["unique_sequence_fraction"], 0.5)
        self.assertEqual(metrics["mean_pairwise_hamming_fraction"], 0.0)
        self.assertEqual(metrics["mean_position_entropy_bits"], 0.0)
        self.assertEqual(metrics["polymorphic_position_fraction"], 0.0)
        self.assertEqual(metrics["gc_fraction_mean"], 0.5)

    def test_ambiguous_bases_are_counted_as_a_fifth_symbol(self):
        metrics = compute_population_complexity(["GCNN", "ATNN"])

        self.assertEqual(metrics["mean_pairwise_hamming_fraction"], 0.5)
        self.assertEqual(metrics["mean_position_entropy_bits"], 0.5)
        self.assertEqual(metrics["polymorphic_position_count"], 2)
        self.assertEqual(metrics["polymorphic_position_fraction"], 0.5)
        self.assertEqual(metrics["gc_fraction_mean"], 0.25)
        self.assertEqual(metrics["gc_fraction_std"], 0.25)
        self.assertEqual(metrics["ambiguous_base_fraction"], 0.5)

    def test_single_sequence_has_zero_diversity(self):
        metrics = compute_population_complexity(["AGCN"])

        self.assertEqual(metrics["mean_pairwise_hamming_fraction"], 0.0)
        self.assertEqual(metrics["mean_position_entropy_bits"], 0.0)
        self.assertEqual(metrics["polymorphic_position_count"], 0)
        self.assertEqual(metrics["unique_sequence_fraction"], 1.0)

    def test_frame_adapter_supports_custom_column(self):
        frame = pl.DataFrame({"insertion": ["AAAA", "AAAT"]})

        metrics = compute_frame_population_complexity(
            frame,
            sequence_column="insertion",
        )

        self.assertEqual(metrics["mean_pairwise_hamming_fraction"], 0.25)

    def test_invalid_populations_fail_clearly(self):
        cases = (
            ([], ValueError, "at least one"),
            ([""], ValueError, "must not be empty"),
            (["AAAA", "AAA"], ValueError, "expected 4"),
            (["AAAX"], ValueError, "unsupported symbols"),
            (["AAAA", None], TypeError, "must be a string"),
        )
        for population, error, message in cases:
            with self.subTest(population=population):
                with self.assertRaisesRegex(error, message):
                    compute_population_complexity(population)

    def test_frame_adapter_validates_column_type_and_nulls(self):
        with self.assertRaisesRegex(KeyError, "region_sequence"):
            compute_frame_population_complexity(pl.DataFrame({"other": ["A"]}))
        with self.assertRaisesRegex(TypeError, "must be String"):
            compute_frame_population_complexity(
                pl.DataFrame({"region_sequence": [1]})
            )
        with self.assertRaisesRegex(ValueError, "must not contain nulls"):
            compute_frame_population_complexity(
                pl.DataFrame(
                    {"region_sequence": ["A", None]},
                    schema={"region_sequence": pl.String},
                )
            )


if __name__ == "__main__":
    unittest.main()

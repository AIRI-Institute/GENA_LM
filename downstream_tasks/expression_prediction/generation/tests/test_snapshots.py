"""Population snapshot callback tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from utils.snapshots import PopulationSnapshotWriter


class _Optimizer:
    def __init__(self, population: pl.DataFrame) -> None:
        self.population = population
        self.calls = 0

    def population_snapshot(self) -> pl.DataFrame:
        self.calls += 1
        return self.population.clone()


class PopulationSnapshotWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.population = pl.DataFrame(
            {
                "candidate_id": ["best", "other"],
                "region_sequence": ["AAAA", "AAAT"],
                "rank": [1, 2],
                "generation_born": [0, 1],
                "source": ["reference", "offspring"],
                "parent_ids": [[], ["best"]],
                "fitness": [3.0, 1.0],
                "on_target__on": [4.0, 2.0],
                "off_target__off": [0.5, 1.0],
            }
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_only_configured_generations_are_written(self):
        optimizer = _Optimizer(self.population)
        writer = PopulationSnapshotWriter.from_config(
            self.root,
            {"generations": [2]},
        )

        writer(optimizer, 1)
        self.assertEqual(optimizer.calls, 0)
        self.assertFalse((self.root / "snapshots").exists())

        writer(optimizer, 2)
        snapshot_dir = self.root / "snapshots/generation_000002"
        self.assertEqual(optimizer.calls, 1)
        self.assertEqual(
            {path.name for path in writer.written_paths},
            {
                "population.parquet",
                "population.csv",
                "insertion_sequences.fasta",
                "metrics.json",
            },
        )
        saved = pl.read_parquet(snapshot_dir / "population.parquet")
        self.assertTrue(saved.equals(self.population))
        self.assertNotIn("sequence", saved.columns)
        self.assertIn("AAAA", (snapshot_dir / "insertion_sequences.fasta").read_text())

        metrics = json.loads((snapshot_dir / "metrics.json").read_text())
        self.assertEqual(metrics["generation"], 2)
        self.assertEqual(metrics["optimization"]["best_candidate_id"], "best")
        self.assertEqual(metrics["optimization"]["best_min_on_target"], 4.0)
        self.assertEqual(metrics["optimization"]["best_max_off_target"], 0.5)
        self.assertEqual(metrics["complexity"]["unique_sequence_count"], 2)
        self.assertEqual(
            metrics["complexity"]["mean_pairwise_hamming_fraction"],
            0.25,
        )

    def test_empty_or_plasmid_population_is_rejected(self):
        writer = PopulationSnapshotWriter(self.root, [0])
        with self.assertRaisesRegex(RuntimeError, "population is empty"):
            writer(_Optimizer(pl.DataFrame()), 0)

        with self.assertRaisesRegex(ValueError, "plasmid"):
            writer(
                _Optimizer(self.population.with_columns(pl.lit("TAAAAG").alias("sequence"))),
                0,
            )

    def test_generation_zero_with_only_empty_parent_lists_writes_csv(self):
        population = self.population.with_columns(
            pl.Series("parent_ids", [[], []])
        )
        self.assertEqual(population.schema["parent_ids"], pl.List(pl.Null))
        writer = PopulationSnapshotWriter(self.root, [0])

        writer(_Optimizer(population), 0)

        saved = pl.read_csv(
            self.root / "snapshots/generation_000000/population.csv"
        )
        self.assertEqual(saved["parent_ids"].fill_null("").to_list(), ["", ""])


if __name__ == "__main__":
    unittest.main()

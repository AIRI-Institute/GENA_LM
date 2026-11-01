"""Result artifact contract tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from gena_expression import AnnotatedSequence, Feature

from utils.config import RunConfig
from utils.results import (
    csv_safe,
    fasta_lines,
    generation_min_on_max_off,
    insertion_only_table,
    write_fasta,
    write_results,
    write_table_pair,
)

from tests.helpers import write_config


class _Optimizer:
    """Public SequenceOptimizer result surface used by the writer."""

    def __init__(self) -> None:
        self.region = Feature("promoter", 1, 5)
        self.evaluations = pl.DataFrame(
            {
                "candidate_id": ["candidate_a", "candidate_b"],
                "generation_born": [0, 1],
                "source": ["reference", "offspring"],
                "parent_ids": [[], ["candidate_a"]],
                "region_sequence": ["AAAA", "CCCC"],
                "sequence": ["TAAAAG", "TCCCCG"],
                "fitness": [1.0, 2.0],
            }
        )
        self.final_population = pl.DataFrame(
            {
                "rank": [1],
                "candidate_id": ["candidate_b"],
                "generation_born": [1],
                "source": ["offspring"],
                "parent_ids": [["candidate_a"]],
                "region_sequence": ["CCCC"],
                "sequence": ["TCCCCG"],
                "fitness": [2.0],
            }
        )
        self.history = pl.DataFrame(
            {
                "generation": [0, 1],
                "best_candidate_id": ["candidate_a", "candidate_b"],
                "best_min_on_target": [1.0, 2.0],
                "best_max_off_target": [0.5, 0.25],
                "best_fitness": [0.5, 1.75],
            }
        )

    @staticmethod
    def summary():
        return {"best_fitness": 2.0}


class ResultTests(unittest.TestCase):
    """Verify the preserved tables and sequence exports."""

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_small_writer_helpers(self):
        frame = pl.DataFrame({"parent_ids": [["a", "b"]], "value": [1]})
        self.assertEqual(csv_safe(frame)["parent_ids"].to_list(), ["a;b"])
        empty_parents = pl.DataFrame({"parent_ids": [[], []], "value": [1, 2]})
        self.assertEqual(empty_parents.schema["parent_ids"], pl.List(pl.Null))
        self.assertEqual(
            csv_safe(empty_parents)["parent_ids"].to_list(),
            ["", ""],
        )
        plain = pl.DataFrame({"value": [1]})
        self.assertTrue(csv_safe(plain).equals(plain))
        with_plasmid = pl.DataFrame(
            {"region_sequence": ["AAAA"], "sequence": ["TAAAAG"]}
        )
        self.assertEqual(
            insertion_only_table(with_plasmid).columns,
            ["region_sequence"],
        )
        self.assertTrue(insertion_only_table(plain).equals(plain))

        paths = write_table_pair(frame, self.root / "table")
        self.assertEqual([path.suffix for path in paths], [".parquet", ".csv"])
        self.assertTrue(all(path.exists() for path in paths))

        records = list(fasta_lines([("a header", "A" * 81)]))
        self.assertEqual(records, [">a_header\n", "A" * 80 + "\n", "A\n"])
        fasta = write_fasta(self.root / "one.fasta", [("one", "ACGT")])
        self.assertEqual(fasta.read_text(), ">one\nACGT\n")

        compact = generation_min_on_max_off(_Optimizer().history)
        self.assertEqual(
            compact.columns,
            [
                "generation",
                "candidate_id",
                "fitness",
                "min_on_target",
                "max_off_target",
            ],
        )
        self.assertEqual(
            compact.row(1, named=True),
            {
                "generation": 1,
                "candidate_id": "candidate_b",
                "fitness": 1.75,
                "min_on_target": 2.0,
                "max_off_target": 0.25,
            },
        )

    def test_complete_result_contract_and_dynamic_region_length(self):
        config = RunConfig.load(write_config(self.root))
        output = self.root / "written"
        output.mkdir()
        samples = (
            AnnotatedSequence("AAAA", name="reference", metadata={"chrom": "chr1"}),
            AnnotatedSequence("CCCC", name="initial", metadata={"chrom": "chr2"}),
        )

        written = write_results(
            _Optimizer(),
            output,
            config=config,
            sampled_population=samples,
        )

        self.assertEqual(len(written), 16)
        self.assertTrue(all(path.exists() for path in written))
        expected_original = {
            "evaluations.parquet",
            "evaluations.csv",
            "final_population.parquet",
            "final_population.csv",
            "history.parquet",
            "history.csv",
            "generation_min_on_max_off.parquet",
            "generation_min_on_max_off.csv",
            "summary.json",
            "run_config.json",
            "output_manifest.json",
            "evaluated_region_sequences.fasta",
            "final_population_region_sequences.fasta",
            "best_insertion_sequence.fasta",
        }
        self.assertTrue(expected_original.issubset({path.name for path in written}))
        self.assertNotIn(
            "sequence",
            pl.read_parquet(output / "final_population.parquet").columns,
        )
        self.assertNotIn(
            "sequence",
            pl.read_csv(output / "evaluations.csv").columns,
        )
        self.assertFalse((output / "final_population_full_sequences.fasta").exists())
        self.assertFalse((output / "best_full_sequence.fasta").exists())
        self.assertIn("CCCC", (output / "best_insertion_sequence.fasta").read_text())

        manifest = json.loads((output / "output_manifest.json").read_text())
        self.assertEqual(manifest["sequence_exports"]["region_length"], 4)
        self.assertFalse(
            manifest["sequence_exports"]["contains_materialized_plasmids"]
        )
        self.assertTrue(
            manifest["generation_min_on_max_off_semantics"][
                "same_candidate_required"
            ]
        )
        self.assertNotIn("600", json.dumps(manifest))
        metadata = json.loads((output / "initial_population_metadata.json").read_text())
        self.assertEqual(metadata[0]["optimizer_role"], "reference")
        self.assertEqual(metadata[1]["optimizer_role"], "initial")
        initial_fasta = (output / "initial_population.fasta").read_text()
        self.assertIn("role=reference", initial_fasta)
        self.assertIn("role=initial", initial_fasta)


if __name__ == "__main__":
    unittest.main()

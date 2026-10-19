"""Genome sampling, target loading, and plasmid replacement tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from gena_expression import AnnotatedSequence, Feature

from utils.inputs import (
    build_genome,
    build_plasmid_template,
    build_promoter_sampler,
    collect_promoter_population,
    load_target_group,
    load_targets,
)

from tests.helpers import write_plasmid


class _InfiniteSamples:
    """Simple deterministic sampler test double."""

    def __init__(self, samples):
        self.samples = list(samples)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        sample = self.samples[self.index % len(self.samples)]
        self.index += 1
        return sample


class InputTests(unittest.TestCase):
    """Verify genome-only sampling and exact plasmid replacement."""

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_target_groups_are_loaded_from_separate_directories(self):
        on_dir = self.root / "on"
        off_dir = self.root / "off"
        on_dir.mkdir()
        off_dir.mkdir()
        (on_dir / "on.json").write_text(json.dumps({"id": "ON", "cell": "brain"}))
        (off_dir / "off.json").write_text(json.dumps({"id": "OFF", "cell": "pancreas"}))
        common = {"seed": 7, "strict": True, "search_fields": ["id"]}
        on_group = {"directory": on_dir, "keys": ["ON"], "filters": {}, "key_filters": {}}
        off_group = {"directory": off_dir, "keys": ["OFF"], "filters": {}, "key_filters": {}}

        on, off = load_targets({**common, "on_target": on_group, "off_target": off_group})

        self.assertEqual(list(on), ["ON"])
        self.assertEqual(list(off), ["OFF"])
        self.assertEqual(on["ON"].description["cell"], "brain")
        self.assertEqual(off["OFF"].description["cell"], "pancreas")
        self.assertEqual(load_target_group({**on_group, "keys": []}, common), {})

    def test_genome_and_sampler_factories_forward_configuration(self):
        genome_config = {
            "fasta_path": Path("genome.fa"),
            "annotation_path": Path("genes.gtf"),
            "build": "test",
            "chrom_style": "chr",
            "cache": False,
        }
        fake_genome = object()
        with patch("utils.inputs.Genome", return_value=fake_genome) as genome_cls:
            self.assertIs(build_genome(genome_config), fake_genome)
        genome_cls.assert_called_once_with(
            fasta_path=Path("genome.fa"),
            annotation=Path("genes.gtf"),
            build="test",
            chrom_style="chr",
            cache=False,
        )

        sampler_config = {
            "seq_len": 4,
            "tss_sampling_window": 10,
            "seed": 3,
            "transcript_feature_types": ["transcript"],
            "include_features": False,
        }
        fake_sampler = object()
        with patch("utils.inputs.PromoterSampler", return_value=fake_sampler) as sampler_cls:
            self.assertIs(build_promoter_sampler(fake_genome, sampler_config), fake_sampler)
        sampler_cls.assert_called_once_with(genome=fake_genome, **sampler_config)

    def test_population_collection_is_length_checked_and_can_deduplicate(self):
        samples = [
            AnnotatedSequence("AAAA", name="one"),
            AnnotatedSequence("AAAA", name="duplicate"),
            AnnotatedSequence("CCCC", name="two"),
        ]
        collected = collect_promoter_population(
            _InfiniteSamples(samples),
            population_size=2,
            seq_len=4,
            unique_sequences=True,
            max_attempts=4,
        )
        self.assertEqual([sample.sequence for sample in collected], ["AAAA", "CCCC"])

        with self.assertRaisesRegex(RuntimeError, "Collected 1"):
            collect_promoter_population(
                _InfiniteSamples([AnnotatedSequence("AAAA")]),
                population_size=2,
                seq_len=4,
                unique_sequences=True,
                max_attempts=2,
            )
        with self.assertRaisesRegex(ValueError, "expected 4"):
            collect_promoter_population(
                _InfiniteSamples([AnnotatedSequence("AAA")]),
                population_size=1,
                seq_len=4,
                unique_sequences=False,
                max_attempts=1,
            )
        with self.assertRaisesRegex(TypeError, "AnnotatedSequence"):
            collect_promoter_population(
                _InfiniteSamples(["AAAA"]),
                population_size=1,
                seq_len=4,
                unique_sequences=False,
                max_attempts=1,
            )

    def _plasmid_config(self, *, fasta: bool = False) -> dict:
        genbank, fasta_path = write_plasmid(self.root)
        return {
            "genbank_path": genbank,
            "fasta_path": fasta_path if fasta else None,
            "reporter_feature_name": "luc2",
            "replacement_feature_name": "replace_me",
            "region_feature_name": "promoter",
            "tokenization_center_name": "reporter_tss",
            "circular": True,
            "window_size": "full",
        }

    def test_genome_sample_is_pasted_by_replacing_plasmid_feature(self):
        config = self._plasmid_config(fasta=True)
        sample = AnnotatedSequence(
            "TGCATG",
            name="tx1|chr1:10-16(+)",
            features=(Feature("tss", 2, 3, type="tss"),),
            metadata={"chrom": "chr1", "genomic_start": 10, "genomic_end": 16},
        )

        template = build_plasmid_template(config, sample)

        promoter = template.feature("promoter")
        self.assertEqual(template.sequence[promoter.start:promoter.end], "TGCATG")
        self.assertEqual([f.name for f in template.features].count("promoter"), 1)
        self.assertEqual([f.name for f in template.features].count("reporter_tss"), 1)
        self.assertNotIn("tss", [f.name for f in template.features])
        self.assertEqual(template.metadata["reference_promoter_metadata"]["chrom"], "chr1")
        self.assertEqual(len(template), 22)

    def test_plasmid_input_mismatches_and_feature_ambiguity_fail(self):
        config = self._plasmid_config(fasta=True)
        Path(config["fasta_path"]).write_text(">test\nTTTTTTTTTTTTTTTTTTTT\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_plasmid_template(config, AnnotatedSequence("AAAA"))

        ambiguous = self._plasmid_config()
        genbank, _ = write_plasmid(self.root, duplicate_target=True)
        ambiguous["genbank_path"] = genbank
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_plasmid_template(ambiguous, AnnotatedSequence("AAAA"))

        overlap = {**self._plasmid_config(), "replacement_feature_name": "luc2"}
        with self.assertRaisesRegex(ValueError, "overlaps the reporter TSS"):
            build_plasmid_template(overlap, AnnotatedSequence("AAAA"))

    def test_existing_center_name_is_rejected(self):
        config = self._plasmid_config()
        config["tokenization_center_name"] = "luc2"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            build_plasmid_template(config, AnnotatedSequence("AAAA"))


if __name__ == "__main__":
    unittest.main()

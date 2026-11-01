"""Configuration validation and path-resolution tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.config import ConfigError, RunConfig, json_ready, resource_value

from tests.helpers import complete_config, write_config


class ConfigTests(unittest.TestCase):
    """Verify explicit settings and config-relative path handling."""

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_load_resolves_paths_and_resources(self):
        config = RunConfig.load(write_config(self.root))

        self.assertEqual(config.source_path, self.root / "config.yaml")
        self.assertEqual(config.section("plasmid")["genbank_path"], self.root / "plasmids/test.gb")
        self.assertEqual(
            resource_value(config.section("model")["dna_tokenizer"], label="dna"),
            str(self.root / "dna_tokenizer"),
        )
        self.assertEqual(
            resource_value(config.section("model")["description_tokenizer"], label="description"),
            "description/model",
        )
        self.assertEqual(
            config.section("model")["model_cls"],
            f"{self.root / 'model.py'}::Model",
        )
        exported = config.as_dict()
        self.assertEqual(exported["config_source"], str(self.root / "config.yaml"))
        self.assertIsInstance(exported["genome"]["fasta_path"], str)

    def test_section_is_a_defensive_copy_and_unknown_section_fails(self):
        config = RunConfig.load(write_config(self.root))
        section = config.section("sampler")
        section["seq_len"] = 999
        self.assertEqual(config.section("sampler")["seq_len"], 4)
        with self.assertRaisesRegex(ConfigError, "Unknown configuration section"):
            config.section("missing")

    def test_missing_and_inconsistent_values_fail(self):
        cases = []
        missing_section = complete_config(self.root)
        missing_section.pop("runtime")
        cases.append((missing_section, "Missing required section"))

        missing_key = complete_config(self.root)
        missing_key["plasmid"].pop("replacement_feature_name")
        cases.append((missing_key, "plasmid.replacement_feature_name"))

        short_window = complete_config(self.root)
        short_window["sampler"]["tss_sampling_window"] = 3
        cases.append((short_window, "at least sampler.seq_len"))

        no_targets = complete_config(self.root)
        no_targets["conditions"]["on_target"]["keys"] = []
        no_targets["conditions"]["off_target"]["keys"] = []
        cases.append((no_targets, "At least one"))

        bad_resource = complete_config(self.root)
        bad_resource["model"]["dna_tokenizer"] = {"path": "a", "identifier": "b"}
        cases.append((bad_resource, "exactly one"))

        for index, (data, message) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(ConfigError, message):
                RunConfig.load(write_config(self.root, data))

    def test_resource_and_json_helpers(self):
        self.assertIsNone(resource_value(None, label="resource"))
        self.assertEqual(resource_value("hub/id", label="resource"), "hub/id")
        self.assertEqual(json_ready({"path": Path("a"), "items": (Path("b"),)}), {
            "path": "a",
            "items": ["b"],
        })
        with self.assertRaisesRegex(ConfigError, "usable resource"):
            resource_value({}, label="resource")

    def test_zero_prefetch_is_a_valid_explicit_runtime_choice(self):
        data = complete_config(self.root)
        data["runtime"]["prefetch_batches"] = 0
        config = RunConfig.load(write_config(self.root, data))
        self.assertEqual(config.section("runtime")["prefetch_batches"], 0)

    def test_snapshot_generations_are_validated_and_sorted(self):
        data = complete_config(self.root)
        data["snapshots"]["generations"] = [1, 0]
        config = RunConfig.load(write_config(self.root, data))
        self.assertEqual(config.section("snapshots")["generations"], [0, 1])

        invalid_cases = [
            ([0, 0], "duplicates"),
            ([-1], "non-negative"),
            ([2], "must not exceed"),
            ("0,1", "must be a list"),
        ]
        for generations, message in invalid_cases:
            with self.subTest(generations=generations):
                data = complete_config(self.root)
                data["snapshots"]["generations"] = generations
                with self.assertRaisesRegex(ConfigError, message):
                    RunConfig.load(write_config(self.root, data))

    def test_initial_population_size_is_validated(self):
        smaller_than_selected = complete_config(self.root)
        smaller_than_selected["optimization"]["initial_population_size"] = 2
        with self.assertRaisesRegex(
            ConfigError,
            "initial_population_size must be at least",
        ):
            RunConfig.load(write_config(self.root, smaller_than_selected))

        too_few_attempts = complete_config(self.root)
        too_few_attempts["sampler"]["max_attempts"] = 4
        with self.assertRaisesRegex(
            ConfigError,
            "max_attempts must be at least",
        ):
            RunConfig.load(write_config(self.root, too_few_attempts))


if __name__ == "__main__":
    unittest.main()

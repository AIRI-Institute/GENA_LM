"""Component-factory tests."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from gena_expression import (
    ExpressionScorer,
    LinearCombinationScorer,
    MinOnTargetMaxOffTargetFitness,
    ParallelCrossoverMutationGenerator,
    SequentialCrossoverMutationGenerator,
    TrackFeatureScorer,
    TrackWindowScorer,
)

from utils.components import build_fitness, build_model, build_offspring_generator, build_scorer
from utils.config import ConfigError


class ComponentTests(unittest.TestCase):
    """Verify supported declarative component choices."""

    def test_model_factory_forwards_every_model_setting(self):
        config = {
            "genalm_home": Path("/tmp/genalm"),
            "model_cls": "module::Model",
            "checkpoint": Path("checkpoint.bin"),
            "config": Path("model.yaml"),
            "dna_tokenizer": {"path": Path("dna")},
            "description_tokenizer": {"identifier": "hub/description"},
            "dna_max_seq_len": 32,
            "desc_max_seq_len": 16,
            "token_len_for_fetch": 4,
            "num_before": 8,
            "device": "cpu",
            "output_names": {"expression": 0},
            "description_formatter": "pkg::format_description",
        }
        sentinel = object()
        with patch("utils.components.SequenceModel.load", return_value=sentinel) as load:
            self.assertIs(build_model(config), sentinel)
        self.assertEqual(os.environ["GENALM_HOME"], str(Path("/tmp/genalm")))
        load.assert_called_once_with(
            model_cls="module::Model",
            checkpoint=Path("checkpoint.bin"),
            config=Path("model.yaml"),
            dna_tokenizer="dna",
            description_tokenizer="hub/description",
            dna_max_seq_len=32,
            desc_max_seq_len=16,
            token_len_for_fetch=4,
            num_before=8,
            device="cpu",
            output_names={"expression": 0},
            description_formatter="pkg::format_description",
        )

    def test_scorer_factories(self):
        self.assertIsInstance(build_scorer({"type": "expression", "params": {}}), ExpressionScorer)
        self.assertIsInstance(
            build_scorer({"type": "track_feature", "params": {"features": "promoter"}}),
            TrackFeatureScorer,
        )
        self.assertIsInstance(
            build_scorer({"type": "track_window", "params": {"center": "reporter_tss", "width_bp": 5}}),
            TrackWindowScorer,
        )
        combined = build_scorer(
            {
                "type": "linear_combination",
                "params": {
                    "name": "combined",
                    "components": {
                        "expression": {
                            "coefficient": 1.0,
                            "scorer": {"type": "expression", "params": {}},
                        }
                    },
                },
            }
        )
        self.assertIsInstance(combined, LinearCombinationScorer)

        with self.assertRaisesRegex(ConfigError, "Unsupported scorer"):
            build_scorer({"type": "unknown", "params": {}})
        with self.assertRaisesRegex(ConfigError, "non-empty components"):
            build_scorer({"type": "linear_combination", "params": {"components": {}}})

    def test_fitness_and_offspring_factories(self):
        self.assertIsNone(build_fitness({"type": "mean_on_minus_mean_off", "params": {}}))
        self.assertIsInstance(
            build_fitness(
                {"type": "min_on_max_off", "params": {"on_power": 1, "off_power": 2}}
            ),
            MinOnTargetMaxOffTargetFitness,
        )
        with self.assertRaisesRegex(ConfigError, "does not accept"):
            build_fitness({"type": "mean_on_minus_mean_off", "params": {"x": 1}})
        with self.assertRaisesRegex(ConfigError, "Unsupported fitness"):
            build_fitness({"type": "unknown", "params": {}})

        common = {"crossover_probability": 0.5, "mutations_per_child": 1, "random_seed": 1}
        self.assertIsInstance(
            build_offspring_generator({"type": "sequential", "params": common}),
            SequentialCrossoverMutationGenerator,
        )
        parallel = build_offspring_generator(
            {"type": "parallel", "params": {**common, "max_workers": 1, "batches_per_worker": 1}}
        )
        self.assertIsInstance(parallel, ParallelCrossoverMutationGenerator)
        parallel.close()
        with self.assertRaisesRegex(ConfigError, "Unsupported offspring"):
            build_offspring_generator({"type": "unknown", "params": {}})


if __name__ == "__main__":
    unittest.main()

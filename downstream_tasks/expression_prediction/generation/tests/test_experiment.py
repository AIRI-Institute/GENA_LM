"""Optimizer assembly and top-level orchestration tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from gena_expression import AnnotatedSequence, Condition, ExpressionScorer, Feature, Prediction

from utils.config import RunConfig
from utils.experiment import build_optimizer, prepare_output_directory, run_experiment

from tests.helpers import write_config


class _Offspring:
    """Minimal record-mode offspring generator."""

    def __call__(self, parents, context):
        del parents, context
        return []


class _ExpressionModel:
    """Deterministic public-model test double for a real optimizer run."""

    @staticmethod
    def predict_multiple_sequences(sequences, conditions=None, **kwargs):
        del kwargs
        return [
            Prediction(
                sequence=sequence,
                condition=condition,
                logits=None,
                outputs={"expression": float(sequence.sequence.count("G"))},
            )
            for sequence, condition in zip(sequences, conditions or ())
        ]


class ExperimentTests(unittest.TestCase):
    """Verify the boundary between sampled promoters and plasmid optimization."""

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = RunConfig.load(write_config(self.root))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_output_directory_protection(self):
        output_config = self.config.section("output")
        output = prepare_output_directory(output_config)
        self.assertTrue(output.is_dir())
        (output / "existing.txt").write_text("occupied")
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            prepare_output_directory(output_config)
        output_config["allow_existing"] = True
        self.assertEqual(prepare_output_directory(output_config), output)

    def test_optimizer_uses_sample_zero_in_plasmid_and_rest_as_initial_population(self):
        template = AnnotatedSequence(
            "TAAAAG",
            features=(
                Feature("promoter", 1, 5, type="optimization_region"),
                Feature("reporter_tss", 5, 6, type="tokenization_center"),
            ),
        )
        samples = (
            AnnotatedSequence("AAAA", name="reference"),
            AnnotatedSequence("CCCC", name="initial_1"),
            AnnotatedSequence("GGGG", name="initial_2"),
        )
        optimizer = build_optimizer(
            self.config,
            template=template,
            sampled_population=samples,
            on_targets={"ON": "on"},
            off_targets={"OFF": "off"},
            scorer=ExpressionScorer(),
            fitness=None,
            offspring_generator=_Offspring(),
        )

        self.assertEqual(template.sequence[1:5], samples[0].sequence)
        self.assertEqual(optimizer.initial_population, ("CCCC", "GGGG"))
        self.assertEqual(optimizer.region.name, "promoter")
        self.assertEqual(optimizer.tokenization_center, "reporter_tss")

    def test_synthetic_optimizer_run_scores_all_genome_sampled_seeds(self):
        template = AnnotatedSequence(
            "TAAAAG",
            features=(
                Feature("promoter", 1, 5, type="optimization_region"),
                Feature("reporter_tss", 5, 6, type="tokenization_center"),
            ),
        )
        samples = tuple(AnnotatedSequence(value) for value in ("AAAA", "CCCC", "GGGG"))
        optimizer = build_optimizer(
            self.config,
            template=template,
            sampled_population=samples,
            on_targets={"ON": Condition("on", "on")},
            off_targets={"OFF": Condition("off", "off")},
            scorer=ExpressionScorer(),
            fitness=None,
            offspring_generator=_Offspring(),
        ).run_evolution(_ExpressionModel(), show_progress=False)

        self.assertEqual(optimizer.evaluations.height, 3)
        self.assertEqual(
            set(optimizer.evaluations["region_sequence"].to_list()),
            {"AAAA", "CCCC", "GGGG"},
        )
        reference = optimizer.evaluations.filter(
            optimizer.evaluations["source"] == "reference"
        ).row(0, named=True)
        self.assertEqual(reference["region_sequence"], samples[0].sequence)

    def test_run_experiment_closes_resources_and_writes_after_evolution(self):
        fake_genome = Mock()
        fake_sampler = object()
        samples = (
            AnnotatedSequence("AAAA", name="reference"),
            AnnotatedSequence("CCCC", name="initial_1"),
            AnnotatedSequence("GGGG", name="initial_2"),
        )
        template = AnnotatedSequence("TAAAAG")
        offspring = Mock()
        optimizer = Mock()
        paths = [self.root / "results/run/summary.json"]

        with (
            patch("utils.experiment.load_targets", return_value=({"ON": "on"}, {"OFF": "off"})),
            patch("utils.experiment.build_genome", return_value=fake_genome),
            patch("utils.experiment.build_promoter_sampler", return_value=fake_sampler),
            patch("utils.experiment.collect_promoter_population", return_value=samples) as collect,
            patch("utils.experiment.build_plasmid_template", return_value=template) as plasmid,
            patch("utils.experiment.build_scorer", return_value="scorer"),
            patch("utils.experiment.build_fitness", return_value="fitness"),
            patch("utils.experiment.build_offspring_generator", return_value=offspring),
            patch("utils.experiment.build_model", return_value="model"),
            patch("utils.experiment.build_optimizer", return_value=optimizer) as optimizer_builder,
            patch("utils.experiment.write_results", return_value=paths) as writer,
        ):
            result = run_experiment(self.config)

        self.assertEqual(result, paths)
        fake_genome.close.assert_called_once_with()
        collect.assert_called_once_with(
            fake_sampler,
            population_size=3,
            seq_len=4,
            unique_sequences=True,
            max_attempts=20,
        )
        plasmid.assert_called_once_with(self.config.section("plasmid"), samples[0])
        optimizer_builder.assert_called_once()
        optimizer.run_evolution.assert_called_once_with(
            model="model",
            **self.config.section("runtime"),
        )
        offspring.close.assert_called_once_with()
        writer.assert_called_once()


if __name__ == "__main__":
    unittest.main()

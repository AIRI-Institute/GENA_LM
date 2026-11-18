"""Tests for fixed-length evolutionary sequence optimization."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import gena_expression as gx


class _ExpressionModel:
    """Small deterministic model implementing the public batch method."""

    def __init__(self) -> None:
        self.calls = []

    def predict_multiple_sequences(
        self,
        sequences,
        conditions=None,
        **kwargs,
    ):
        """Return expression proportional to region G count and condition."""

        sequence_list = list(sequences)
        condition_list = list(conditions or ())
        self.calls.append(
            {
                "rows": len(sequence_list),
                "return_tokens": kwargs["return_tokens"],
                "retention": kwargs["retention"],
            }
        )
        predictions = []
        for sequence, condition in zip(sequence_list, condition_list):
            g_count = sequence.sequence[:3].count("G")
            multiplier = 2.0 if condition.name == "on" else 0.25
            predictions.append(
                gx.Prediction(
                    sequence=sequence,
                    condition=condition,
                    logits=None,
                    outputs={"expression": multiplier * g_count},
                )
            )
        return predictions


class SequenceOptimizerTests(unittest.TestCase):
    """Verify optimization state, scoring, and generator interfaces."""

    @staticmethod
    def _sequence() -> gx.AnnotatedSequence:
        """Return one sequence with annotations inside and outside the region."""

        return gx.AnnotatedSequence(
            "AAACCC",
            name="test",
            features=(
                gx.Feature("promoter", 0, 3),
                gx.Feature("tss", 3, 4),
            ),
            metadata={"source": "test"},
        )

    @staticmethod
    def _targets():
        """Return one on-target and one off-target condition."""

        return (
            {"on": gx.Condition("on", "on condition")},
            {"off": gx.Condition("off", "off condition")},
        )

    @staticmethod
    def _mutate_first_a(parents):
        """Replace the first available A in each parent with G."""

        for parent in parents:
            position = parent.find("A")
            if position >= 0:
                yield parent[:position] + "G" + parent[position + 1 :]

    def _optimizer(self, *, selection_strategy="top_k"):
        """Return a small deterministic optimizer."""

        on_targets, off_targets = self._targets()
        return gx.SequenceOptimizer(
            self._sequence(),
            region="promoter",
            tokenization_center="tss",
            scorer=gx.ExpressionScorer(),
            on_targets=on_targets,
            off_targets=off_targets,
            offspring_generator=self._mutate_first_a,
            population_size=2,
            offspring_size=4,
            n_generations=3,
            selection_strategy=selection_strategy,
            elite_count=1,
            random_seed=7,
        )

    def test_expression_scorer_returns_absolute_expression(self):
        condition = gx.Condition("on", "on condition")
        prediction = gx.Prediction(
            sequence=self._sequence(),
            condition=condition,
            logits=None,
            outputs={"expression": 3.0},
        )

        result = gx.ExpressionScorer(transform="log2").score_prediction(
            prediction
        )

        self.assertEqual(result.name, "expression")
        self.assertEqual(result.score, 2.0)
        self.assertEqual(result.provenance["mode"], "absolute")

    def test_min_on_max_off_fitness_supports_configurable_powers(self):
        fitness = gx.MinOnTargetMaxOffTargetFitness(
            on_power=1.0,
            off_power=2.0,
            off_weight=0.5,
        )

        value = fitness(
            {"on_a": 5.0, "on_b": 3.0},
            {"off_a": 2.0, "off_b": 1.0},
        )

        self.assertEqual(value, 1.0)

    def test_run_evolution_builds_expected_public_tables(self):
        model = _ExpressionModel()
        optimizer = self._optimizer().run_evolution(
            model,
            evaluation_chunk_size=2,
            show_progress=False,
        )

        self.assertEqual(optimizer.final_population.height, 2)
        self.assertEqual(optimizer.evaluations.height, 4)
        self.assertEqual(optimizer.history.height, 4)
        self.assertEqual(
            optimizer.final_population.row(0, named=True)["region_sequence"],
            "GGG",
        )
        self.assertEqual(
            optimizer.final_population.row(0, named=True)["fitness"],
            5.25,
        )
        self.assertIn("on_target__on", optimizer.final_population.columns)
        self.assertIn("off_target__off", optimizer.final_population.columns)
        self.assertEqual(
            optimizer.history.columns,
            [
                "generation",
                "proposed_candidates",
                "evaluated_candidates",
                "population_size",
                "best_candidate_id",
                "best_fitness",
                "mean_fitness",
                "best_min_on_target",
                "best_max_off_target",
                "mean_min_on_target",
                "mean_max_off_target",
            ],
        )
        generation_two = optimizer.history.row(2, named=True)
        generation_two_best = next(
            row
            for row in optimizer.evaluations.iter_rows(named=True)
            if row["candidate_id"] == generation_two["best_candidate_id"]
        )
        self.assertEqual(generation_two_best["region_sequence"], "GGA")
        self.assertEqual(
            generation_two["best_fitness"],
            generation_two_best["fitness"],
        )
        self.assertEqual(generation_two["best_min_on_target"], 4.0)
        self.assertEqual(generation_two["best_max_off_target"], 0.5)
        self.assertEqual(
            generation_two["best_min_on_target"],
            generation_two_best["on_target__on"],
        )
        self.assertEqual(
            generation_two["best_max_off_target"],
            generation_two_best["off_target__off"],
        )
        self.assertEqual(generation_two["mean_min_on_target"], 3.0)
        self.assertEqual(generation_two["mean_max_off_target"], 0.375)
        self.assertTrue(
            all(not call["return_tokens"] for call in model.calls)
        )
        self.assertTrue(
            all(call["retention"] == "scalars" for call in model.calls)
        )

    def test_progress_reports_one_step_per_post_selection_generation(self):
        class _Progress:
            def __init__(self) -> None:
                self.updates = 0
                self.descriptions = []
                self.postfixes = []
                self.closed = False

            def set_description_str(self, value):
                self.descriptions.append(value)

            def set_postfix(self, value, refresh=False):
                self.postfixes.append(value)

            def update(self, value):
                self.updates += value

            def close(self):
                self.closed = True

        progress = _Progress()
        with patch("tqdm.auto.tqdm", return_value=progress) as tqdm:
            self._optimizer().run_evolution(
                _ExpressionModel(),
                evaluation_chunk_size=1,
                show_progress=True,
            )

        tqdm.assert_called_once()
        self.assertEqual(progress.updates, 4)
        self.assertEqual(len(progress.postfixes), 4)
        self.assertEqual(progress.postfixes[-1]["generation"], "3/3")
        self.assertTrue(progress.closed)

    def test_generation_callback_receives_insertion_only_population(self):
        snapshots = []

        def collect(optimizer, generation):
            snapshots.append((generation, optimizer.population_snapshot()))

        optimizer = self._optimizer().run_evolution(
            _ExpressionModel(),
            show_progress=False,
            on_generation_complete=collect,
        )

        self.assertEqual([generation for generation, _ in snapshots], [0, 1, 2, 3])
        self.assertEqual([frame.height for _, frame in snapshots], [1, 2, 2, 2])
        self.assertTrue(all("sequence" not in frame.columns for _, frame in snapshots))
        self.assertTrue(
            all("region_sequence" in frame.columns for _, frame in snapshots)
        )
        self.assertEqual(
            snapshots[-1][1]["candidate_id"].to_list(),
            optimizer.final_population["candidate_id"].to_list(),
        )

    def test_generation_callback_must_be_callable(self):
        with self.assertRaisesRegex(TypeError, "on_generation_complete"):
            self._optimizer().run_evolution(
                _ExpressionModel(),
                show_progress=False,
                on_generation_complete="not callable",
            )

    def test_restored_population_continues_with_absolute_generations(self):
        contexts = []
        callbacks = []

        def generate(parents, context):
            contexts.append(context["generation"])
            yield {
                "region_sequence": "GGA",
                "parent_ids": [parents[0]["candidate_id"]],
            }

        on_targets, off_targets = self._targets()
        optimizer = gx.SequenceOptimizer(
            self._sequence(),
            region="promoter",
            tokenization_center="tss",
            scorer=gx.ExpressionScorer(),
            on_targets=on_targets,
            off_targets=off_targets,
            offspring_generator=generate,
            offspring_generator_mode="records",
            population_size=2,
            offspring_size=1,
            n_generations=4,
            initial_population=["GAA"],
            initial_generation=2,
        ).run_evolution(
            _ExpressionModel(),
            show_progress=False,
            on_generation_complete=lambda _, generation: callbacks.append(
                generation
            ),
        )

        self.assertEqual(callbacks, [2, 3, 4])
        self.assertEqual(contexts, [3, 4])
        self.assertEqual(optimizer.history["generation"].to_list(), [2, 3, 4])
        restored = optimizer.evaluations.filter(
            optimizer.evaluations["source"] == "restored"
        )
        self.assertEqual(restored.height, 2)
        self.assertEqual(restored["generation_born"].unique().to_list(), [2])

    def test_initial_generation_is_validated(self):
        on_targets, off_targets = self._targets()
        common = {
            "region": "promoter",
            "tokenization_center": "tss",
            "scorer": gx.ExpressionScorer(),
            "on_targets": on_targets,
            "off_targets": off_targets,
            "offspring_generator": self._mutate_first_a,
            "n_generations": 3,
        }
        with self.assertRaisesRegex(TypeError, "initial_generation"):
            gx.SequenceOptimizer(
                self._sequence(), initial_generation=True, **common
            )
        with self.assertRaisesRegex(ValueError, "between 0 and n_generations"):
            gx.SequenceOptimizer(
                self._sequence(), initial_generation=4, **common
            )

    def test_top_k_warm_restart_matches_uninterrupted_generations(self):
        on_targets, off_targets = self._targets()
        saved = []

        def make_optimizer(sequence, initial_population, initial_generation):
            return gx.SequenceOptimizer(
                sequence,
                region="promoter",
                tokenization_center="tss",
                scorer=gx.ExpressionScorer(),
                on_targets=on_targets,
                off_targets=off_targets,
                offspring_generator=gx.SequentialCrossoverMutationGenerator(
                    mutations_per_child=1,
                    random_seed=19,
                ),
                offspring_generator_mode="records",
                population_size=2,
                offspring_size=2,
                n_generations=4,
                initial_population=initial_population,
                initial_generation=initial_generation,
                selection_strategy="top_k",
            )

        uninterrupted = make_optimizer(
            self._sequence(),
            ["AAC"],
            0,
        ).run_evolution(
            _ExpressionModel(),
            show_progress=False,
            on_generation_complete=lambda optimizer, generation: (
                saved.append(optimizer.population_snapshot())
                if generation == 2
                else None
            ),
        )
        restored_regions = saved[0]["region_sequence"].to_list()
        restored_sequence = gx.AnnotatedSequence(
            restored_regions[0] + "CCC",
            features=self._sequence().features,
        )
        resumed = make_optimizer(
            restored_sequence,
            restored_regions[1:],
            2,
        ).run_evolution(_ExpressionModel(), show_progress=False)

        self.assertEqual(
            resumed.final_population["candidate_id"].to_list(),
            uninterrupted.final_population["candidate_id"].to_list(),
        )
        self.assertEqual(
            resumed.final_population["fitness"].to_list(),
            uninterrupted.final_population["fitness"].to_list(),
        )

    def test_all_selection_strategies_keep_elite_and_finish(self):
        for strategy in ("top_k", "tournament", "rank_weighted"):
            with self.subTest(strategy=strategy):
                optimizer = self._optimizer(
                    selection_strategy=strategy
                ).run_evolution(
                    _ExpressionModel(),
                    show_progress=False,
                )
                best = optimizer.top(1).row(0, named=True)
                self.assertEqual(best["region_sequence"], "GGG")
                self.assertTrue(math.isfinite(best["fitness"]))
                evaluation_by_id = {
                    row["candidate_id"]: row
                    for row in optimizer.evaluations.iter_rows(named=True)
                }
                for history_row in optimizer.history.iter_rows(named=True):
                    champion = evaluation_by_id[
                        history_row["best_candidate_id"]
                    ]
                    self.assertEqual(
                        history_row["best_fitness"], champion["fitness"]
                    )
                    self.assertEqual(
                        history_row["best_min_on_target"],
                        champion["on_target__on"],
                    )
                    self.assertEqual(
                        history_row["best_max_off_target"],
                        champion["off_target__off"],
                    )

    def test_record_generator_receives_context_without_support_types(self):
        contexts = []

        def generate(parents, context):
            contexts.append(context)
            yield {
                "region_sequence": "GAA",
                "parent_ids": [parents[0]["candidate_id"]],
            }

        on_targets, off_targets = self._targets()
        optimizer = gx.SequenceOptimizer(
            self._sequence(),
            region=(0, 3),
            tokenization_center="tss",
            scorer=gx.ExpressionScorer(),
            on_targets=on_targets,
            off_targets=off_targets,
            offspring_generator=generate,
            offspring_generator_mode="records",
            population_size=1,
            offspring_size=1,
            n_generations=1,
            random_seed=11,
        ).run_evolution(
            _ExpressionModel(),
            show_progress=False,
        )

        self.assertEqual(contexts[0]["generation"], 1)
        self.assertEqual(contexts[0]["region"]["length"], 3)
        self.assertIsInstance(contexts[0]["rng"], __import__("random").Random)
        self.assertEqual(
            len(optimizer.final_population.row(0, named=True)["parent_ids"]),
            1,
        )

    def test_materialization_preserves_annotations_and_coordinate_map(self):
        optimizer = self._optimizer().run_evolution(
            _ExpressionModel(),
            show_progress=False,
        )

        best = optimizer.best_sequence

        self.assertEqual(best.sequence, "GGGCCC")
        self.assertEqual(best.features, optimizer.sequence.features)
        self.assertIs(best.coordinate_map, optimizer.sequence.coordinate_map)
        self.assertEqual(best.metadata["source"], "test")
        self.assertIn("optimization_candidate_id", best.metadata)

    def test_reset_clears_results_and_allows_a_repeat_run(self):
        optimizer = self._optimizer().run_evolution(
            _ExpressionModel(),
            show_progress=False,
        )

        optimizer.reset()

        self.assertTrue(optimizer.evaluations.is_empty())
        self.assertTrue(optimizer.final_population.is_empty())
        optimizer.run_evolution(_ExpressionModel(), show_progress=False)
        self.assertEqual(
            optimizer.final_population.row(0, named=True)["region_sequence"],
            "GGG",
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for package-level and granular retention behavior."""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import gena_expression as gx
from gena_expression.conditions import Condition
from gena_expression.inference.outputs import (
    PairPrediction,
    Prediction,
    retain_prediction,
)
from gena_expression.inference import SequenceModel
from gena_expression.scoring.results import ScoringResult
from gena_expression.scoring import VariantInterpreter


class RetentionTests(unittest.TestCase):
    """Verify presets, propagation, compaction, and explicit failures."""

    def setUp(self) -> None:
        """Save the caller's active context-local policy."""

        self.previous_policy = gx.get_retention_policy()

    def tearDown(self) -> None:
        """Restore the policy so tests remain isolated."""

        gx.set_retention_policy(self.previous_policy)

    @staticmethod
    def _condition() -> Condition:
        return Condition(name="K562", description="cell type is K562.")

    def test_default_and_context_manager(self):
        """Full mode is the default and temporary modes restore correctly."""

        gx.set_retention_mode("full")
        self.assertEqual(gx.get_retention_policy(), gx.FULL_RETENTION)
        with gx.retention_mode("scalars"):
            self.assertEqual(gx.get_retention_policy(), gx.SCALAR_RETENTION)
        self.assertEqual(gx.get_retention_policy(), gx.FULL_RETENTION)

    def test_scalar_prediction_keeps_value_and_drops_heavy_fields(self):
        """Scalar predictions retain usable numeric outputs without payloads."""

        prediction = Prediction(
            sequence=SimpleNamespace(name="sequence"),
            condition=self._condition(),
            logits=[1.25],
            outputs={"expression": [1.25], "logits": [1.25]},
            tokens=SimpleNamespace(tokens=[]),
            provenance={"checkpoint": "model.ckpt"},
            description_tokens={"input_ids": [1, 2, 3]},
        )
        compact = retain_prediction(prediction, gx.SCALAR_RETENTION.prediction)

        self.assertEqual(compact.scalar(), 1.25)
        self.assertIsNone(compact.sequence)
        self.assertIsNone(compact.logits)
        self.assertIsNone(compact.tokens)
        self.assertIsNone(compact.provenance)
        self.assertIsNone(compact.description_tokens)
        self.assertNotIn("logits", compact.outputs)
        with self.assertRaises(gx.RetainedDataError):
            compact.track()

    def test_package_mode_propagates_to_model_boundary(self):
        """A package preset is resolved once and passed to inference internals."""

        model = SequenceModel.__new__(SequenceModel)
        calls = []

        model._broadcast_conditions = MethodType(
            lambda self, count, conditions, condition: [condition] * count,
            model,
        )

        @contextmanager
        def preprocessing_executor(self, backend, workers):
            yield None

        def predict_tasks(self, tasks, conditions, **kwargs):
            calls.append(kwargs)
            return [task[0] for task in tasks]

        model._preprocessing_executor = MethodType(preprocessing_executor, model)
        model._predict_sequence_tasks = MethodType(predict_tasks, model)
        model._to_expression_prediction = MethodType(lambda self, value: value, model)

        gx.set_retention_mode("scalars")
        pair = SimpleNamespace(ref="ref", alt="alt")
        returned = model.predict_multiple_pairs(
            [pair],
            condition=self._condition(),
            center=0,
            show_progress=False,
        )

        self.assertEqual(calls[0]["retention"], gx.SCALAR_RETENTION.prediction)
        self.assertEqual(calls[1]["retention"], gx.SCALAR_RETENTION.prediction)
        self.assertIsNone(returned[0].pair)

    def test_scoring_uses_full_inputs_then_returns_compact_result(self):
        """Scorers see full tensors while the returned result is compacted."""

        condition = self._condition()
        variant = SimpleNamespace(
            id="variant-1",
            chrom="chr1",
            pos=9,
            ref="A",
            alt="G",
        )
        sequence_pair = SimpleNamespace(variant=variant, ref="ref", alt="alt")
        ref = Prediction("ref", condition, [1.0], {"expression": [1.0]})
        alt = Prediction("alt", condition, [2.0], {"expression": [2.0]})
        pair_prediction = PairPrediction(ref, alt, sequence_pair, condition)
        calls = []

        class FakeModel:
            def predict_multiple_pairs(self, pairs, **kwargs):
                calls.append(kwargs)
                return [pair_prediction]

        class FakeScorer:
            name = "expression_delta"

            def score(self, prediction):
                self.seen_prediction = prediction
                return ScoringResult(
                    name=self.name,
                    score=prediction.delta_scalar(),
                    prediction=prediction,
                    ref_track=[1.0],
                    alt_track=[2.0],
                    delta_track=[1.0],
                    features=[{"key": "promoter", "score": 1.0, "rows": [1, 2]}],
                    provenance={"scorer": "fake"},
                )

        scorer = FakeScorer()
        result = VariantInterpreter(FakeModel())._score_pairs(
            [sequence_pair],
            [condition],
            ["variant-1"],
            scorer=scorer,
            center=0,
            grouping="no_grouping",
            retention=gx.SCALAR_RETENTION,
            show_progress=False,
        )[0]

        self.assertIs(scorer.seen_prediction, pair_prediction)
        self.assertEqual(calls[0]["retention"], gx.FULL_RETENTION)
        self.assertEqual(result.score, 1.0)
        self.assertIsNone(result.prediction)
        self.assertIsNone(result.ref_track)
        self.assertIsNone(result.alt_track)
        self.assertIsNone(result.delta_track)
        self.assertEqual(result.features, [{"key": "promoter", "score": 1.0}])
        self.assertIsNone(result.provenance)
        payload = json.loads(result.to_json())
        self.assertEqual(payload["variant"]["id"], "variant-1")
        self.assertEqual(payload["condition"], {"name": "K562"})

    def test_custom_policy_can_keep_only_delta_tracks(self):
        """Granular settings can retain a delta track without allele tracks."""

        policy = gx.RetentionPolicy(
            prediction=gx.PredictionRetention(logits=False, tokens=False),
            scoring=gx.ScoringRetention(
                prediction=False,
                tracks="delta",
                features="none",
                provenance=False,
            ),
        )
        self.assertEqual(policy.scoring.tracks, "delta")
        self.assertFalse(policy.prediction.logits)


if __name__ == "__main__":
    unittest.main()

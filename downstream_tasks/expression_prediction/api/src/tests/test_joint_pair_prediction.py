"""Focused tests for joint reference/alternative prediction."""

from __future__ import annotations

import logging
import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from gena_expression.conditions import Condition
from gena_expression.inference import SequenceModel
from gena_expression.scoring import VariantInterpreter


class JointPairPredictionTests(unittest.TestCase):
    """Verify pair ordering, forward boundaries, and public option forwarding."""

    @staticmethod
    def _public_model():
        """Build a lightweight model double for testing public pair orchestration."""

        model = SequenceModel.__new__(SequenceModel)
        calls = []

        def broadcast_conditions(self, row_count, conditions, condition):
            """Return one condition per requested pair."""

            if conditions is not None:
                return list(conditions)
            return [condition] * row_count

        @contextmanager
        def preprocessing_executor(self, backend, workers):
            """Provide the no-worker context expected by the public method."""

            yield None

        def predict_tasks(self, tasks, conditions, **kwargs):
            """Record one logical prediction stage and preserve row identity."""

            calls.append((list(tasks), list(conditions), kwargs))
            return [task[0] for task in tasks]

        def identity_prediction(self, prediction):
            """Keep fake predictions unchanged during pair reconstruction."""

            return prediction

        model._broadcast_conditions = MethodType(broadcast_conditions, model)
        model._preprocessing_executor = MethodType(preprocessing_executor, model)
        model._predict_sequence_tasks = MethodType(predict_tasks, model)
        model._to_expression_prediction = MethodType(identity_prediction, model)
        return model, calls

    @staticmethod
    def _planner_model():
        """Build a model double that exercises the real forward-batch planner."""

        model = SequenceModel.__new__(SequenceModel)
        model.logger = logging.getLogger(__name__)
        model.provenance = {}
        calls = {"pair": [], "condition": []}

        def tokenize_description(self, condition):
            """Return a minimal cached description payload."""

            return {"description": condition.text()}

        def run_pairs(tokenized_batch, desc_encodings):
            """Record one pair-aware forward and return row-aligned logits."""

            names = [tokenized.source.name for tokenized in tokenized_batch]
            calls["pair"].append(names)
            return SimpleNamespace(logits=names)

        def run_condition(tokenized_batch, desc_encoded):
            """Record one repeated-condition forward and return row-aligned logits."""

            names = [tokenized.source.name for tokenized in tokenized_batch]
            calls["condition"].append(names)
            return SimpleNamespace(logits=names)

        def logits_to_cpu(model_output):
            """Return fake logits without tensor conversion."""

            return model_output.logits

        def expression_from_logits(row_logits):
            """Use the fake row logit as the expression payload."""

            return row_logits

        def attach_description(prediction, desc_encoded):
            """Skip token attachment for the minimal fake description."""

            return prediction

        def clear_cuda_cache():
            """Avoid importing torch in the lightweight test environment."""

        model.tokenize_description = MethodType(tokenize_description, model)
        model._run_tokenized_pairs = run_pairs
        model._run_tokenized_repeated_condition = run_condition
        model._logits_to_cpu = logits_to_cpu
        model._expression_from_logits = expression_from_logits
        model._attach_description_tokens = attach_description
        model.clear_cuda_cache = clear_cuda_cache
        return model, calls

    @staticmethod
    def _tokenized_rows(pair_count):
        """Return interleaved fake reference and alternative token rows."""

        rows = []
        for pair_idx in range(pair_count):
            rows.extend(
                [
                    SimpleNamespace(source=SimpleNamespace(name=f"ref_{pair_idx}")),
                    SimpleNamespace(source=SimpleNamespace(name=f"alt_{pair_idx}")),
                ]
            )
        return rows

    def test_public_joint_execution_interleaves_alleles_once(self):
        """Joint mode should call the sequence pipeline once with atomic pairs."""

        model, calls = self._public_model()
        condition = Condition(name="K562", description="cell type is K562.")
        pairs = [
            SimpleNamespace(ref="ref_0", alt="alt_0"),
            SimpleNamespace(ref="ref_1", alt="alt_1"),
        ]

        predictions = model.predict_multiple_pairs(
            pairs,
            condition=condition,
            center=0,
            grouping="no_grouping",
            pair_execution="joint",
            max_pairs_per_forward=2,
            show_progress=False,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual([task[0] for task in calls[0][0]], ["ref_0", "alt_0", "ref_1", "alt_1"])
        self.assertTrue(calls[0][2]["paired_rows"])
        self.assertEqual(calls[0][2]["max_pairs_per_forward"], 2)
        self.assertEqual([(row.ref, row.alt) for row in predictions], [("ref_0", "alt_0"), ("ref_1", "alt_1")])

    def test_public_separate_execution_preserves_two_pass_protocol(self):
        """Separate mode should retain reference-first and alternative-second calls."""

        model, calls = self._public_model()
        condition = Condition(name="K562", description="cell type is K562.")
        pairs = [
            SimpleNamespace(ref="ref_0", alt="alt_0"),
            SimpleNamespace(ref="ref_1", alt="alt_1"),
        ]

        model.predict_multiple_pairs(
            pairs,
            condition=condition,
            center=0,
            pair_execution="separate",
            max_records_per_forward=2,
            show_progress=False,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual([task[0] for task in calls[0][0]], ["ref_0", "ref_1"])
        self.assertEqual([task[0] for task in calls[1][0]], ["alt_0", "alt_1"])
        self.assertEqual(calls[0][2]["max_records_per_forward"], 2)

    def test_pair_forward_uses_b_by_two_description_layout(self):
        """The physical model call should receive B=P, N=2 description blocks."""

        class FakeTensor:
            """Track tensor shapes through unsqueeze and repeat operations."""

            def __init__(self, shape):
                """Store an immutable fake tensor shape."""

                self.shape = tuple(shape)

            def unsqueeze(self, dimension):
                """Insert one dimension using positive indexing."""

                shape = list(self.shape)
                shape.insert(dimension, 1)
                return FakeTensor(shape)

            def repeat(self, *repeats):
                """Multiply every dimension by its repeat count."""

                return FakeTensor([size * repeat for size, repeat in zip(self.shape, repeats)])

        class FakeTorch:
            """Provide the small torch surface used by the pair runner."""

            bool = "bool"
            bfloat16 = "bfloat16"

            @staticmethod
            def zeros(shape, dtype, device):
                """Return a zero tensor represented only by its shape."""

                return FakeTensor(shape)

            @staticmethod
            @contextmanager
            def autocast(**kwargs):
                """Provide a no-op autocast context."""

                yield

            @staticmethod
            @contextmanager
            def no_grad():
                """Provide a no-op inference context."""

                yield

        model = SequenceModel.__new__(SequenceModel)
        model.device = "cpu"
        calls = []

        def stack_dna(tokenized_sequences):
            """Return a four-row DNA batch with a fixed test length."""

            return FakeTensor((len(tokenized_sequences), 10)), FakeTensor((len(tokenized_sequences), 10))

        def stack_descriptions(desc_encodings):
            """Return one description row for each input encoding."""

            return FakeTensor((len(desc_encodings), 5)), FakeTensor((len(desc_encodings), 5))

        def fake_model(**kwargs):
            """Record the final tensors passed to the expression model."""

            calls.append(kwargs)
            return SimpleNamespace(logits=None)

        model._stack_tokenized_dna = stack_dna
        model._stack_description_encodings = stack_descriptions
        model.model = fake_model
        with patch.dict("sys.modules", {"torch": FakeTorch}):
            model._run_tokenized_pairs([object()] * 4, [object()] * 4)

        self.assertEqual(calls[0]["input_ids"].shape, (4, 10))
        self.assertEqual(calls[0]["desc_input_ids"].shape, (2, 2, 5))
        self.assertEqual(calls[0]["dataset_flag"].shape, (2, 2))

    def test_no_grouping_chunks_only_at_pair_boundaries(self):
        """Pair-aware no-grouping batches should have B=P and N=2."""

        model, calls = self._planner_model()
        rows = self._tokenized_rows(3)
        conditions = [
            condition
            for pair_idx in range(3)
            for condition in (
                Condition(name=f"condition_{pair_idx}", description=f"condition {pair_idx}"),
                Condition(name=f"condition_{pair_idx}", description=f"condition {pair_idx}"),
            )
        ]

        predictions = model._predict_multiple_tokenized_sequences(
            rows,
            conditions,
            grouping="no_grouping",
            preprocessing_workers=0,
            preprocessing_backend="thread",
            max_records_per_forward=None,
            prefetch_batches=0,
            show_progress=False,
            batch_method="test_joint_pairs",
            paired_rows=True,
            max_pairs_per_forward=2,
        )

        self.assertEqual(calls["pair"], [["ref_0", "alt_0", "ref_1", "alt_1"], ["ref_2", "alt_2"]])
        self.assertEqual(
            [prediction.sequence.name for prediction in predictions],
            ["ref_0", "alt_0", "ref_1", "alt_1", "ref_2", "alt_2"],
        )
        self.assertEqual(predictions[0].provenance["dataset_flag_shape"], (2, 2))
        self.assertEqual(predictions[4].provenance["dataset_flag_shape"], (1, 2))

    def test_condition_grouping_keeps_complete_pairs(self):
        """Repeated-condition batching should chunk by pairs, not physical rows."""

        model, calls = self._planner_model()
        rows = self._tokenized_rows(3)
        condition = Condition(name="K562", description="cell type is K562.")
        predictions = model._predict_multiple_tokenized_sequences(
            rows,
            [condition] * len(rows),
            grouping="condition",
            preprocessing_workers=0,
            preprocessing_backend="thread",
            max_records_per_forward=None,
            prefetch_batches=0,
            show_progress=False,
            batch_method="test_joint_condition",
            paired_rows=True,
            max_pairs_per_forward=2,
        )

        self.assertEqual(calls["condition"], [["ref_0", "alt_0", "ref_1", "alt_1"], ["ref_2", "alt_2"]])
        self.assertEqual(predictions[0].provenance["dataset_flag_shape"], (1, 4))
        self.assertEqual(predictions[4].provenance["dataset_flag_shape"], (1, 2))

    def test_interpreter_forwards_joint_options(self):
        """The interpreter's shared scoring path should pass pair options through."""

        calls = []

        class FakeModel:
            """Record calls made by the interpreter."""

            def predict_multiple_pairs(self, pairs, **kwargs):
                """Capture pair-prediction keyword arguments."""

                calls.append(kwargs)
                return []

        scorer = SimpleNamespace(name="expression", score=None)
        interpreter = VariantInterpreter(FakeModel())
        interpreter._score_pairs(
            [],
            [],
            [],
            scorer=scorer,
            center=0,
            grouping="no_grouping",
            pair_execution="joint",
            max_pairs_per_forward=3,
            show_progress=False,
        )

        self.assertEqual(calls[0]["pair_execution"], "joint")
        self.assertEqual(calls[0]["max_pairs_per_forward"], 3)


if __name__ == "__main__":
    unittest.main()

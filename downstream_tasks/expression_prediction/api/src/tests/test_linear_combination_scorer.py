"""Tests for weighted scalar scorer composition."""

from __future__ import annotations

import unittest

import gena_expression as gx


class LinearCombinationScorerTests(unittest.TestCase):
    """Verify absolute score composition and optimizer-facing metadata."""

    def test_combines_named_absolute_scorers(self):
        prediction = gx.Prediction(
            sequence=gx.AnnotatedSequence("ACGT", name="test"),
            condition=gx.Condition("condition", "test condition"),
            logits=None,
            outputs={"first": 2.0, "second": 3.0},
        )
        scorer = gx.LinearCombinationScorer(
            {
                "first": (gx.ExpressionScorer(output="first"), 0.5),
                "second": (gx.ExpressionScorer(output="second"), 2.0),
            }
        )

        result = scorer.score_prediction(prediction)

        self.assertEqual(result.score, 7.0)
        self.assertFalse(scorer.requires_tokens)
        self.assertEqual(
            result.provenance["components"]["second"]["contribution"],
            6.0,
        )

    def test_rejects_non_scalar_component_scores(self):
        prediction = gx.Prediction(
            sequence=gx.AnnotatedSequence("ACGT", name="test"),
            condition=gx.Condition("condition", "test condition"),
            logits=None,
            outputs={"multiple": [1.0, 2.0]},
        )
        scorer = gx.LinearCombinationScorer(
            {
                "multiple": (
                    gx.ExpressionScorer(output="multiple"),
                    1.0,
                )
            }
        )

        with self.assertRaisesRegex(TypeError, "scalar component"):
            scorer.score_prediction(prediction)


if __name__ == "__main__":
    unittest.main()

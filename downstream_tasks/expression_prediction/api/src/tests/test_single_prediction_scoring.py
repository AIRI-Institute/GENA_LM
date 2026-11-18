"""Tests for absolute scoring of one sequence prediction."""

from __future__ import annotations

import unittest

import gena_expression as gx
from gena_expression.conditions import Condition
from gena_expression.inference.outputs import PairPrediction
from gena_expression.sequences import AnnotatedSequence, Feature, SequencePair
from gena_expression.tracks import TrackPrediction


class _TrackOnlyPrediction:
    """Minimal prediction object exposing sequence metadata and one track."""

    def __init__(self, sequence, condition, values, tokens):
        self.sequence = sequence
        self.condition = condition
        self.values = list(values)
        self.tokens = list(tokens)

    def track(self, name="atac", channel=0):
        """Return the configured synthetic coordinate-aware track."""

        return TrackPrediction(
            name=name,
            values=self.values,
            tokens=self.tokens,
            channel=channel,
        )


class SinglePredictionScoringTests(unittest.TestCase):
    """Verify absolute scores without changing reference/alternative scores."""

    @staticmethod
    def _sequence(name: str) -> AnnotatedSequence:
        """Return one annotated sequence used by all synthetic predictions."""

        return AnnotatedSequence(
            "A" * 60,
            name=name,
            features=(
                Feature(name="left", start=0, end=10),
                Feature(name="right", start=40, end=50),
            ),
        )

    @staticmethod
    def _tokens():
        """Return six non-overlapping 10-bp token intervals."""

        return [
            {
                "input_position": index + 1,
                "start": index * 10,
                "end": (index + 1) * 10,
                "center": index * 10 + 5,
            }
            for index in range(6)
        ]

    @classmethod
    def _prediction(cls):
        """Return a single prediction with track values one through six."""

        condition = Condition(name="K562", description="cell type is K562.")
        return _TrackOnlyPrediction(
            cls._sequence("single"),
            condition,
            [1, 2, 3, 4, 5, 6],
            cls._tokens(),
        )

    @classmethod
    def _pair_prediction(cls):
        """Return a pair whose alternative track is twice its reference track."""

        condition = Condition(name="K562", description="cell type is K562.")
        ref_sequence = cls._sequence("ref")
        alt_sequence = cls._sequence("alt")
        return PairPrediction(
            ref=_TrackOnlyPrediction(
                ref_sequence,
                condition,
                [1, 2, 3, 4, 5, 6],
                cls._tokens(),
            ),
            alt=_TrackOnlyPrediction(
                alt_sequence,
                condition,
                [2, 4, 6, 8, 10, 12],
                cls._tokens(),
            ),
            pair=SequencePair(ref=ref_sequence, alt=alt_sequence),
            condition=condition,
        )

    def test_track_window_scores_one_absolute_track(self):
        scorer = gx.TrackWindowScorer(
            center=30,
            width_bp=20,
            aggregate="sum",
            sign="ref-alt",
        )

        result = scorer.score_prediction(self._prediction())

        self.assertIsInstance(result, gx.PredictionScoringResult)
        self.assertEqual(result.score, 70.0)
        self.assertEqual(result.score_window, gx.ScoreWindow(20, 40, 30))
        self.assertEqual(result.score_windows, (result.score_window,))
        self.assertEqual(result.provenance["mode"], "absolute")

    def test_track_window_resolves_a_named_sequence_feature(self):
        scorer = gx.TrackWindowScorer(
            center="feature:right",
            width_bp=10,
            aggregate="mean",
        )

        result = scorer.score_prediction(self._prediction())

        self.assertEqual(result.score, 5.0)
        self.assertEqual(result.score_window, gx.ScoreWindow(40, 50, 45))

    def test_track_feature_scores_selected_features(self):
        scorer = gx.TrackFeatureScorer(
            features=("left", "right"),
            aggregate="sum",
            sign="ref-alt",
        )

        result = scorer.score_prediction(self._prediction())

        self.assertEqual(result.score, 60.0)
        self.assertEqual(
            result.score_windows,
            (
                gx.ScoreWindow(0, 10, 5),
                gx.ScoreWindow(40, 50, 45),
            ),
        )

    def test_track_all_features_returns_named_absolute_scores(self):
        scorer = gx.TrackAllFeaturesScorer(
            aggregate="sum",
            sign="ref-alt",
            feature_source="alt",
        )

        result = scorer.score_prediction(self._prediction())

        self.assertEqual(result.score, {"left": 10.0, "right": 50.0})
        self.assertEqual(
            [entry["key"] for entry in result.features],
            ["left", "right"],
        )
        self.assertEqual(len(result.score_windows), 2)

    def test_scorer_set_delegates_and_reports_single_prediction_scores(self):
        scorer = gx.ScorerSet(
            {
                "window": gx.TrackWindowScorer(center=30, width_bp=20),
                "features": gx.TrackFeatureScorer(features=("left", "right")),
                "all": gx.TrackAllFeaturesScorer(),
            }
        )

        report = scorer.score_prediction(self._prediction())

        self.assertIsInstance(report, gx.PredictionReport)
        self.assertEqual(
            report.scores(),
            {
                "window": 70.0,
                "features": 60.0,
                "all": {"left": 10.0, "right": 50.0},
            },
        )
        self.assertEqual(report["window"].score, 70.0)

    def test_scorer_set_names_unsupported_single_prediction_scorers(self):
        scorer = gx.ScorerSet(
            {
                "window": gx.TrackWindowScorer(center=30, width_bp=20),
                "expression": gx.ExpressionDeltaScorer(),
            }
        )

        with self.assertRaisesRegex(
            TypeError,
            "without single-prediction support: expression",
        ):
            scorer.score_prediction(self._prediction())

    def test_pair_only_aggregate_is_rejected_for_one_prediction(self):
        scorers = (
            gx.TrackWindowScorer(center=30, width_bp=20, aggregate="max_abs_delta"),
            gx.TrackFeatureScorer(features="left", aggregate="max_abs_delta"),
            gx.TrackAllFeaturesScorer(aggregate="max_abs_delta"),
        )

        for scorer in scorers:
            with self.subTest(scorer=type(scorer).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "requires a reference/alternative pair",
                ):
                    scorer.score_prediction(self._prediction())

    def test_existing_pair_scores_are_unchanged(self):
        prediction = self._pair_prediction()
        scorers = {
            "window": gx.TrackWindowScorer(center=30, width_bp=20),
            "features": gx.TrackFeatureScorer(features=("left", "right")),
            "all": gx.TrackAllFeaturesScorer(),
        }

        self.assertEqual(scorers["window"].score(prediction).score, 70.0)
        self.assertEqual(scorers["features"].score(prediction).score, 60.0)
        self.assertEqual(
            scorers["all"].score(prediction).score,
            {"left": 10.0, "right": 50.0},
        )

        report = gx.ScorerSet(scorers).score(prediction)
        self.assertIsInstance(report, gx.VariantReport)
        self.assertEqual(
            report.scores(),
            {
                "window": 70.0,
                "features": 60.0,
                "all": {"left": 10.0, "right": 50.0},
            },
        )


if __name__ == "__main__":
    unittest.main()

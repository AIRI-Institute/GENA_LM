"""Tests for adaptive peak-centered track-effect scoring."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import gena_expression as gx
from gena_expression.conditions import Condition
from gena_expression.inference.outputs import PairPrediction
from gena_expression.scoring.results import retain_scoring_result
from gena_expression.sequences import AnnotatedSequence, SequencePair
from gena_expression.tracks import TrackPrediction


class _TrackOnlyPrediction:
    """Minimal prediction object exposing one coordinate-aware track."""

    def __init__(self, values, tokens):
        self.values = list(values)
        self.tokens = list(tokens)

    def track(self, name="atac", channel=0):
        return TrackPrediction(
            name=name,
            values=self.values,
            tokens=self.tokens,
            channel=channel,
        )


class TrackEffectPeakScorerTests(unittest.TestCase):
    """Verify selection, aggregation, exports, and retention behavior."""

    @staticmethod
    def _prediction(ref_values, alt_values):
        tokens = [
            {
                "input_position": index + 1,
                "start": index * 10,
                "end": (index + 1) * 10,
                "center": index * 10 + 5,
            }
            for index in range(len(ref_values))
        ]
        length = len(tokens) * 10
        ref_sequence = AnnotatedSequence("A" * length, name="ref")
        alt_sequence = AnnotatedSequence("A" * length, name="alt")
        pair = SequencePair(ref=ref_sequence, alt=alt_sequence)
        condition = Condition(name="K562", description="cell type is K562.")
        return PairPrediction(
            ref=_TrackOnlyPrediction(ref_values, tokens),
            alt=_TrackOnlyPrediction(alt_values, tokens),
            pair=pair,
            condition=condition,
        )

    def test_public_export_and_primary_peak(self):
        prediction = self._prediction(
            [0, 0, 0, 0, 0, 0],
            [0, 10, 0, 9, 0, 8],
        )
        scorer = gx.TrackEffectPeakScorer(
            search_center=30,
            search_width_bp=60,
            window_bp=10,
        )

        result = scorer.score(prediction)

        self.assertEqual(result.score, 100.0)
        self.assertEqual(len(result.features), 1)
        self.assertEqual(result.features[0]["position"], 15)
        self.assertEqual(result.features[0]["candidate_delta"], 10.0)
        self.assertEqual(result.score_window.to_dict()["start"], 10)

    def test_relative_effect_and_minimum_distance_select_multiple_peaks(self):
        prediction = self._prediction(
            [0, 0, 0, 0, 0, 0],
            [0, 10, 0, 9, 0, 8],
        )
        scorer = gx.TrackEffectPeakScorer(
            search_center=30,
            search_width_bp=60,
            window_bp=10,
            max_points=3,
            min_distance_bp=25,
            min_effect_fraction=0.80,
        )

        result = scorer.score(prediction)

        self.assertEqual(result.score, 180.0)
        self.assertEqual(
            [entry["position"] for entry in result.features],
            [15, 55],
        )

    def test_relative_effect_fraction_filters_weaker_peak(self):
        prediction = self._prediction(
            [0, 0, 0, 0, 0, 0],
            [0, 10, 0, 9, 0, 8],
        )
        scorer = gx.TrackEffectPeakScorer(
            search_center=30,
            search_width_bp=60,
            window_bp=10,
            max_points=3,
            min_distance_bp=0,
            min_effect_fraction=0.85,
        )

        result = scorer.score(prediction)

        self.assertEqual(result.score, 190.0)
        self.assertEqual(
            [entry["position"] for entry in result.features],
            [15, 35],
        )

    def test_default_direction_excludes_opposite_effect_peaks(self):
        prediction = self._prediction(
            [0, 0, 0, 0, 0, 0],
            [0, 10, 0, -9, 0, 8],
        )
        common = {
            "search_center": 30,
            "search_width_bp": 60,
            "window_bp": 10,
            "max_points": 3,
            "min_distance_bp": 0,
            "min_effect_fraction": 0.80,
        }

        same_direction = gx.TrackEffectPeakScorer(**common).score(prediction)
        any_direction = gx.TrackEffectPeakScorer(
            **common,
            direction="any",
        ).score(prediction)

        self.assertEqual(
            [entry["position"] for entry in same_direction.features],
            [15, 55],
        )
        self.assertEqual(
            [entry["position"] for entry in any_direction.features],
            [15, 35, 55],
        )

    def test_smoothing_uses_overlap_weighted_bp_window(self):
        prediction = self._prediction(
            [0, 0, 0],
            [0, 10, 0],
        )
        scorer = gx.TrackEffectPeakScorer(
            search_center=15,
            search_width_bp=30,
            window_bp=10,
            smoothing_bp=20,
        )

        result = scorer.score(prediction)

        self.assertEqual(result.features[0]["position"], 15)
        self.assertEqual(result.features[0]["selection_effect"], 5.0)

    def test_overlapping_point_windows_are_aggregated_as_a_union(self):
        prediction = self._prediction(
            [0, 0, 0, 0, 0, 0],
            [0, 10, 1, 9, 0, 0],
        )
        scorer = gx.TrackEffectPeakScorer(
            search_center=30,
            search_width_bp=60,
            window_bp=30,
            max_points=2,
            min_distance_bp=10,
            min_effect_fraction=0.85,
        )

        result = scorer.score(prediction)

        self.assertEqual(result.score, 200.0)
        self.assertEqual(
            result.provenance["ref_windows"],
            [{"start": 0, "end": 50, "center": 25, "units": "sequence"}],
        )
        self.assertEqual(
            [entry["score"] for entry in result.features],
            [110.0, 100.0],
        )

    def test_log2_ratio_and_scalar_retention(self):
        prediction = self._prediction(
            [0, 1, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0],
        )
        scorer = gx.TrackEffectPeakScorer(
            search_center=30,
            search_width_bp=60,
            window_bp=10,
            comparison="log2_ratio",
            pseudocount=1.0,
        )

        full = scorer.score(prediction)
        compact = retain_scoring_result(full, gx.SCALAR_RETENTION)

        self.assertAlmostEqual(
            full.score,
            math.log2(31.0 / 11.0),
        )
        self.assertIsNone(compact.prediction)
        self.assertIsNone(compact.ref_track)
        self.assertEqual(
            compact.features,
            [{"key": "effect_peak#1", "score": full.features[0]["score"]}],
        )

    def test_no_finite_candidate_returns_nan_with_warning(self):
        prediction = self._prediction(
            [0, 0, 0],
            [float("nan"), float("nan"), float("nan")],
        )
        scorer = gx.TrackEffectPeakScorer(
            search_center=15,
            search_width_bp=30,
            window_bp=10,
        )

        result = scorer.score(prediction)

        self.assertTrue(math.isnan(result.score))
        self.assertEqual(result.features, [])
        self.assertIn("No finite effect peak", result.warnings[-1])

    def test_argument_validation(self):
        invalid = (
            {"search_width_bp": 0},
            {"window_bp": 0},
            {"max_points": 0},
            {"min_distance_bp": -1},
            {"min_effect_fraction": 0.0},
            {"min_effect_fraction": 1.1},
            {"smoothing_bp": 0},
            {"selection": "largest"},
            {"direction": "both"},
            {"spatial_aggregate": "max"},
            {"comparison": "ratio"},
            {"combine_points": "sum"},
            {"sign": "absolute"},
            {"comparison": "log2_ratio", "pseudocount": 0.0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gx.TrackEffectPeakScorer(**kwargs)


if __name__ == "__main__":
    unittest.main()

"""Tests for allele-aware multi-window scoring-result metadata and plots."""

from __future__ import annotations

import json
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import gena_expression as gx
from gena_expression.conditions import Condition
from gena_expression.inference.outputs import PairPrediction
from gena_expression.scoring.results import (
    ResultIdentity,
    ScoreWindow,
    ScoringResult,
    retain_scoring_result,
)
from gena_expression.sequences import AnnotatedSequence, Feature, SequencePair
from gena_expression.tracks import TrackPrediction


class _TrackOnlyPrediction:
    """Minimal prediction exposing one coordinate-aware track."""

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


class ScoringResultWindowTests(unittest.TestCase):
    """Verify multi-window metadata without changing numeric score behavior."""

    @staticmethod
    def _rows(value_key: str):
        return [
            {"start": start, "end": start + 10, value_key: float(index + 1)}
            for index, start in enumerate(range(0, 70, 10))
        ]

    @staticmethod
    def _prediction():
        tokens = [
            {
                "input_position": index + 1,
                "start": index * 10,
                "end": (index + 1) * 10,
                "center": index * 10 + 5,
            }
            for index in range(6)
        ]
        features = (
            Feature(name="left", start=0, end=10),
            Feature(name="right", start=40, end=50),
        )
        ref = AnnotatedSequence("A" * 60, name="ref", features=features)
        alt = AnnotatedSequence("A" * 60, name="alt", features=features)
        pair = SequencePair(ref=ref, alt=alt)
        condition = Condition(name="K562", description="cell type is K562.")
        return PairPrediction(
            ref=_TrackOnlyPrediction([1, 2, 3, 4, 5, 6], tokens),
            alt=_TrackOnlyPrediction([2, 4, 6, 8, 10, 12], tokens),
            pair=pair,
            condition=condition,
        )

    @staticmethod
    def _indel_prediction():
        def tokens(bounds):
            return [
                {
                    "input_position": index + 1,
                    "start": start,
                    "end": end,
                    "center": (start + end) // 2,
                }
                for index, (start, end) in enumerate(zip(bounds, bounds[1:]))
            ]

        features = (
            Feature(name="site", start=0, end=10),
            Feature(name="site", start=40, end=50),
        )
        pair = SequencePair(
            ref=AnnotatedSequence("A" * 60, name="ref", features=features),
            alt=AnnotatedSequence("A" * 20 + "GG" + "A" * 40, name="alt"),
        )
        condition = Condition(name="K562", description="cell type is K562.")
        return PairPrediction(
            ref=_TrackOnlyPrediction(
                [1, 2, 3, 4, 5, 6],
                tokens([0, 10, 20, 30, 40, 50, 60]),
            ),
            alt=_TrackOnlyPrediction(
                [2, 4, 100, 8, 16, 32, 64],
                tokens([0, 10, 20, 22, 32, 42, 52, 62]),
            ),
            pair=pair,
            condition=condition,
        )

    @staticmethod
    def _result():
        ref_windows = (
            ScoreWindow(0, 10, 5),
            ScoreWindow(40, 50, 45),
        )
        alt_windows = (
            ScoreWindow(0, 10, 5),
            ScoreWindow(43, 53, 48),
        )
        return ScoringResult(
            name="multi_window",
            score=60.0,
            prediction=None,
            score_window=ScoreWindow(0, 50, 25),
            ref_track=ScoringResultWindowTests._rows("value"),
            alt_track=ScoringResultWindowTests._rows("value"),
            delta_track=ScoringResultWindowTests._rows("delta"),
            identity=ResultIdentity(condition="K562"),
            ref_score_windows=ref_windows,
            alt_score_windows=alt_windows,
        )

    def test_track_plots_mark_each_window_in_the_correct_coordinates(self):
        result = self._result()

        fig, axes = result.plot_tracks()
        try:
            self.assertEqual(
                result._score_windows_for("ref"),
                result.ref_score_windows,
            )
            self.assertEqual(
                result._score_windows_for("alt"),
                result.alt_score_windows,
            )
            self.assertEqual(
                result._score_windows_for("delta"),
                result.ref_score_windows,
            )
            self.assertEqual([len(axis.patches) for axis in axes], [2, 2, 2])
        finally:
            plt.close(fig)

    def test_single_window_results_keep_the_existing_plot_fallback(self):
        window = ScoreWindow(10, 20, 15)
        result = ScoringResult(
            name="single_window",
            score=1.0,
            prediction=None,
            score_window=window,
            ref_track=self._rows("value"),
            alt_track=self._rows("value"),
            delta_track=self._rows("delta"),
        )

        self.assertEqual(result._score_windows_for("ref"), (window,))
        self.assertEqual(result._score_windows_for("alt"), (window,))
        self.assertEqual(result._score_windows_for("delta"), (window,))

        fig, axes = result.plot_tracks()
        try:
            self.assertEqual([len(axis.patches) for axis in axes], [1, 1, 1])
        finally:
            plt.close(fig)

    def test_feature_view_keeps_allele_specific_windows_and_score(self):
        result = self._result()
        result.features = [
            {
                "key": "right",
                "score": 50.0,
                "score_window": result.ref_score_windows[1].to_dict(),
                "ref_score_window": result.ref_score_windows[1].to_dict(),
                "alt_score_window": result.alt_score_windows[1].to_dict(),
            }
        ]

        selected = result.feature("right")

        self.assertEqual(selected.score, 50.0)
        self.assertEqual(selected.score_window, result.ref_score_windows[1])
        self.assertEqual(selected.ref_score_windows, (result.ref_score_windows[1],))
        self.assertEqual(selected.alt_score_windows, (result.alt_score_windows[1],))

    def test_json_and_retention_include_all_windows(self):
        result = self._result()

        payload = json.loads(result.to_json())
        self.assertEqual(
            payload["ref_score_windows"],
            [window.to_dict() for window in result.ref_score_windows],
        )
        self.assertEqual(
            payload["alt_score_windows"],
            [window.to_dict() for window in result.alt_score_windows],
        )

        retained = retain_scoring_result(
            result,
            gx.SCALAR_RETENTION,
        )
        dropped = retain_scoring_result(
            result,
            gx.ScoringRetention(score_window=False),
        )
        self.assertEqual(retained.ref_score_windows, result.ref_score_windows)
        self.assertEqual(retained.alt_score_windows, result.alt_score_windows)
        self.assertIsNone(retained.ref_track)
        self.assertIsNone(retained.alt_track)
        self.assertIsNone(retained.delta_track)
        self.assertIsNone(dropped.score_window)
        self.assertEqual(dropped.ref_score_windows, ())
        self.assertEqual(dropped.alt_score_windows, ())

    def test_multi_window_scorers_keep_expected_numeric_results(self):
        prediction = self._prediction()

        window_result = gx.TrackWindowScorer(
            center=30,
            width_bp=20,
        ).score(prediction)
        feature_result = gx.TrackFeatureScorer(
            features=("left", "right"),
        ).score(prediction)
        all_features_result = gx.TrackAllFeaturesScorer().score(prediction)

        self.assertEqual(window_result.score, 70.0)
        self.assertEqual(feature_result.score, 60.0)
        self.assertEqual(
            all_features_result.score,
            {"left": 10.0, "right": 50.0},
        )
        self.assertEqual(
            feature_result.ref_score_windows,
            (
                ScoreWindow(0, 10, 5),
                ScoreWindow(40, 50, 45),
            ),
        )
        self.assertEqual(
            feature_result.alt_score_windows,
            feature_result.ref_score_windows,
        )

    def test_indel_windows_use_allele_coordinates_without_changing_scores(self):
        prediction = self._indel_prediction()

        feature_result = gx.TrackFeatureScorer(features="site").score(prediction)
        all_features_result = gx.TrackAllFeaturesScorer().score(prediction)

        self.assertEqual(feature_result.score, 280.0)
        self.assertEqual(
            all_features_result.score,
            {"site#1": 10.0, "site#2": 270.0},
        )
        self.assertEqual(
            feature_result.ref_score_windows,
            (
                ScoreWindow(0, 10, 5),
                ScoreWindow(40, 50, 45),
            ),
        )
        self.assertEqual(
            feature_result.alt_score_windows,
            (
                ScoreWindow(0, 10, 5),
                ScoreWindow(42, 52, 47),
            ),
        )
        self.assertEqual(feature_result.score_window, ScoreWindow(0, 52, 26))

        fig, axes = feature_result.plot_tracks()
        try:
            spans = [
                [(patch.get_x(), patch.get_x() + patch.get_width()) for patch in axis.patches]
                for axis in axes
            ]
            self.assertEqual(spans[0], [(0, 10), (40, 50)])
            self.assertEqual(spans[1], [(0, 10), (42, 52)])
            self.assertEqual(spans[2], [(0, 10), (40, 50)])
        finally:
            plt.close(fig)

        delta_fig, delta_axis = feature_result.plot_delta_track()
        try:
            delta_spans = [
                (patch.get_x(), patch.get_x() + patch.get_width())
                for patch in delta_axis.patches
            ]
            self.assertEqual(delta_spans, [(0, 10), (40, 50)])
        finally:
            plt.close(delta_fig)

        summary_fig, summary_axes = feature_result.plot_summary()
        try:
            summary_spans = [
                [
                    (patch.get_x(), patch.get_x() + patch.get_width())
                    for patch in axis.patches
                ]
                for axis in summary_axes[1:]
            ]
            self.assertEqual(
                summary_spans,
                [
                    [(0, 10), (40, 50)],
                    [(0, 10), (42, 52)],
                    [(0, 10), (40, 50)],
                ],
            )
        finally:
            plt.close(summary_fig)


if __name__ == "__main__":
    unittest.main()

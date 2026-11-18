"""Regression tests for plotting results with retained delta tracks."""

import matplotlib.pyplot as plt
import pytest

from gena_expression import AnnotatedSequence, Condition, RetainedDataError, ScoringRetention, SequencePair
from gena_expression.inference.outputs import PairPrediction, Prediction
from gena_expression.scoring.results import ScoreWindow, ScoringResult, retain_scoring_result


@pytest.fixture
def result():
    ref = AnnotatedSequence("ACGT")
    alt = AnnotatedSequence("AGGT")
    condition = Condition("test", "test description")
    prediction = PairPrediction(
        Prediction(ref, condition, None, {}),
        Prediction(alt, condition, None, {}),
        SequencePair(ref, alt),
        condition,
    )
    return ScoringResult(
        name="track",
        score=1.0,
        prediction=prediction,
        score_window=ScoreWindow(1, 3, 2),
        ref_track=[{"start": 1, "end": 3, "value": 1.0}],
        alt_track=[{"start": 1, "end": 3, "value": 2.0}],
        delta_track=[{"start": 1, "end": 3, "delta": 1.0}],
    )


def test_plot_delta_only_result(result):
    compact = retain_scoring_result(result, ScoringRetention(prediction=False, tracks="delta"))
    fig, ax = compact.plot_delta_track()
    try:
        assert len(ax.collections) == 1
        assert ax.collections[0].get_segments()[0].tolist() == [[1.0, 1.0], [3.0, 1.0]]
        assert len(ax.patches) == 1
    finally:
        plt.close(fig)

    with pytest.raises(RetainedDataError, match="all track rows"):
        compact.plot_tracks()


def test_plot_requires_delta_track(result):
    compact = retain_scoring_result(result, ScoringRetention(tracks="none"))
    with pytest.raises(RetainedDataError, match="tracks='delta'"):
        compact.plot_delta_track()


@pytest.mark.parametrize(
    ("keep_prediction", "tracks", "filenames"),
    [
        (False, "delta", ["track.delta.png"]),
        (True, "delta", ["track.variant.png", "track.delta.png"]),
        (True, "all", ["track.variant.png", "track.tracks.png", "track.delta.png"]),
    ],
)
def test_save_plots_uses_available_tracks(result, tmp_path, keep_prediction, tracks, filenames):
    compact = retain_scoring_result(result, ScoringRetention(prediction=keep_prediction, tracks=tracks))
    fig_numbers = set(plt.get_fignums())
    try:
        saved = compact.save_plots(tmp_path)
        assert [path.name for path in saved] == filenames
        assert all(path.stat().st_size > 0 for path in saved)
    finally:
        for number in set(plt.get_fignums()) - fig_numbers:
            plt.close(number)

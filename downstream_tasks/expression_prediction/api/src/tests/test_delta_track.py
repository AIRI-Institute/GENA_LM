"""Regression tests for coordinate-aligned pair prediction tracks."""

from contextlib import nullcontext
from dataclasses import replace
from math import isnan

import pytest
import torch

from gena_expression import (
    AnnotatedSequence,
    Condition,
    PredictionRetention,
    RegressionScorer,
    RetainedDataError,
    RetentionPolicy,
    SequenceModel,
    SequencePair,
    TrackFeatureBuilder,
    TrackWindowScorer,
)
from gena_expression.inference.outputs import (
    PairPrediction,
    Prediction,
    retain_pair_prediction,
    retain_prediction,
)
from gena_expression.inference.tokenization import TokenizedSequence


def _prediction(sequence, track, condition):
    tokens = [
        {"input_position": i, "start": start, "end": end}
        for i, (start, end, _) in enumerate(track, start=1)
    ]
    logits = torch.tensor([0.0, *(value for _, _, value in track)]).reshape(1, -1, 1)
    tokenized = TokenizedSequence(
        input_ids=torch.zeros(len(tokens) + 1, dtype=torch.int64),
        attention_mask=torch.ones(len(tokens) + 1, dtype=torch.int64),
        tokens=tokens,
        source=sequence,
        center=0,
    )
    return Prediction(sequence, condition, logits, {"expression": 0.0}, tokens=tokenized)


def _pair_prediction(ref, alt, ref_track, alt_track):
    pair = SequencePair(AnnotatedSequence(ref), AnnotatedSequence(alt))
    condition = Condition("test", "test description")
    return PairPrediction(
        _prediction(pair.ref, ref_track, condition),
        _prediction(pair.alt, alt_track, condition),
        pair,
        condition,
    )


@pytest.fixture
def mismatched_tokens():
    return _pair_prediction(
        "ACTGAC",
        "ATTGAC",
        [(0, 3, 2), (3, 6, 8)],
        [(0, 1, 2), (1, 4, 12), (4, 6, 5)],
    )


def test_delta_track_uses_weighted_overlaps_and_reference_coordinates(mismatched_tokens):
    delta = mismatched_tokens.delta_track("difference")
    assert delta.name == "difference"
    assert delta.values == pytest.approx([20 / 3, -2 / 3])
    assert [(row["start"], row["end"]) for row in delta.tokens] == [(0, 3), (3, 6)]
    assert [row["alignment_status"] for row in delta.tokens] == ["mapped", "mapped"]

    reversed_delta = mismatched_tokens.delta_track(sign="ref-alt")
    assert reversed_delta.values == pytest.approx([-20 / 3, 2 / 3])

    score = TrackWindowScorer(center=3, width_bp=6, aggregate="max_abs_delta").score(mismatched_tokens)
    assert score.score == pytest.approx(20 / 3)
    assert [row["delta"] for row in score.delta_track] == delta.values


def test_deleted_and_uncovered_intervals_are_distinct():
    prediction = _pair_prediction(
        "ACGTA",
        "AGTA",
        [(0, 1, 1), (1, 2, 20), (2, 4, 2), (4, 5, 5)],
        [(0, 1, 3), (1, 3, 6)],
    )
    delta = prediction.delta_track()
    assert delta.values[0] == 2
    assert isnan(delta.values[1])
    assert delta.values[2] == 4
    assert isnan(delta.values[3])
    assert [row["alignment_status"] for row in delta.tokens] == [
        "mapped", "deleted", "mapped", "uncovered",
    ]
    assert [(row["alt_start"], row["alt_end"]) for row in delta.tokens] == [
        (0, 1), (1, 1), (1, 3), (3, 4),
    ]

    builder = TrackFeatureBuilder(center=2, width_bp=4, bin_size=1, alignment="coordinate")
    assert builder.build(prediction)[0] == [2, 0, 4, 4]


def test_inserted_bases_shift_downstream_alignment():
    prediction = _pair_prediction(
        "ACTGAC",
        "ACGGTGAC",
        [(0, 1, 1), (3, 6, 4)],
        [(0, 2, 2), (2, 4, 100), (4, 5, 7), (5, 8, 6)],
    )
    delta = prediction.delta_track()
    assert delta.values == [1, 2]
    assert [(row["start"], row["end"]) for row in delta.tokens] == [(0, 1), (3, 6)]
    assert [(row["alt_start"], row["alt_end"]) for row in delta.tokens] == [(0, 1), (5, 8)]


def test_legacy_delta_can_run_without_pair_metadata(mismatched_tokens):
    without_pair = replace(mismatched_tokens, pair=None)
    with pytest.raises(RetainedDataError, match="sequence pair"):
        without_pair.delta_track()
    legacy = without_pair.delta_track(alignment="index")
    assert legacy.values == [0, 4]
    assert [(row["start"], row["end"]) for row in legacy.tokens] == [(0, 1), (1, 4)]


def test_compact_pair_retention_keeps_alignment(mismatched_tokens):
    retention = PredictionRetention(sequence=False)
    compact = retain_pair_prediction(mismatched_tokens, retention)
    assert compact.pair is None
    assert compact.ref.sequence is None
    assert compact.delta_track().values == mismatched_tokens.delta_track().values
    assert retain_pair_prediction(compact, retention).delta_track().values == compact.delta_track().values

    no_logits = retain_pair_prediction(compact, replace(retention, logits=False))
    with pytest.raises(RetainedDataError, match="Logits"):
        no_logits.delta_track()


@pytest.mark.parametrize("pair_execution", ["joint", "separate"])
def test_public_pair_prediction_preserves_alignment_without_source_pair(mismatched_tokens, pair_execution):
    model = SequenceModel.__new__(SequenceModel)
    model._preprocessing_executor = lambda *args: nullcontext()
    by_sequence = {
        mismatched_tokens.pair.ref.sequence: mismatched_tokens.ref,
        mismatched_tokens.pair.alt.sequence: mismatched_tokens.alt,
    }

    def predict_tasks(tasks, conditions, **kwargs):
        return [
            retain_prediction(by_sequence[sequence.sequence], kwargs["retention"])
            for sequence, *_ in tasks
        ]

    model._predict_sequence_tasks = predict_tasks
    computed = model.predict_pair(
        mismatched_tokens.pair,
        condition=mismatched_tokens.condition,
        center=0,
        pair_execution=pair_execution,
        retention=RetentionPolicy(prediction=PredictionRetention(sequence=False)),
    )
    assert computed.pair is None
    assert computed.delta_track().values == mismatched_tokens.delta_track().values


def test_regression_features_preserve_legacy_bins_by_default(mismatched_tokens):
    legacy = TrackFeatureBuilder(center=3, width_bp=6, bin_size=2)
    aligned = replace(legacy, alignment="coordinate")
    legacy_values, legacy_metadata = legacy.build(mismatched_tokens)
    aligned_values, aligned_metadata = aligned.build(mismatched_tokens)
    assert legacy_values == [2, 4, 0]
    assert aligned_values == pytest.approx([20 / 3, 3, -2 / 3])
    assert replace(aligned, aggregate="sum").build(mismatched_tokens)[0] == pytest.approx(
        [40 / 3, 6, -4 / 3]
    )
    assert legacy_metadata == aligned_metadata

    class TrainedModel:
        def predict(self, features):
            assert features.tolist() == [[2, 4, 0]]
            return [5.0]

    scorer = RegressionScorer(TrainedModel(), feature_builder=legacy)
    assert scorer.score(mismatched_tokens).score == 5.0

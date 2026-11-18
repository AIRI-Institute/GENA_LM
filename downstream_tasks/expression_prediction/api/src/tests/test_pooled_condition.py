"""Regression coverage for pooled descriptions through public prediction APIs."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from gena_expression import (
    AnnotatedSequence,
    Condition,
    ExpressionDeltaScorer,
    ExpressionPrediction,
    PooledCondition,
    SequenceModel,
    SequencePair,
    VariantInterpreter,
)
from gena_expression.inference.AG_model import AlphaGenomeSequenceModel


class _DnaTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0
    _ids = dict(zip("ACGT", (3, 4, 6, 7)))

    def encode_plus(self, sequence, *, return_offsets_mapping):
        return {
            "input_ids": [self.cls_token_id, *(self._ids[base] for base in sequence), self.sep_token_id],
            "offset_mapping": [(0, 0), *((i, i + 1) for i in range(len(sequence))), (0, 0)],
        }

    def decode(self, token_ids):
        return next(base for base, token_id in self._ids.items() if token_id == token_ids[0])


class _DescriptionTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, description, *, padding, truncation, max_length, return_tensors):
        ids = torch.tensor([[int(character) for character in description]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _InferenceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = []

    def forward(self, *, input_ids, attention_mask, desc_input_ids, desc_attention_mask, dataset_flag, **kwargs):
        scores = (desc_input_ids * desc_attention_mask).sum(dim=-1)
        values = input_ids.float() + 10 * scores.reshape(-1, 1)
        self.forward_calls.append((dataset_flag.shape, input_ids.tolist(), scores.tolist()))
        # Raw logits may include padding positions, including arbitrary values.
        values = values.masked_fill(attention_mask == 0, -1)
        return SimpleNamespace(logits=values.unsqueeze(-1).to(torch.bfloat16))


@pytest.fixture
def model(monkeypatch):
    model = SequenceModel(
        _InferenceModule(),
        _DnaTokenizer(),
        _DescriptionTokenizer(),
        dna_max_seq_len=64,
        desc_max_seq_len=32,
        token_len_for_fetch=1,
        num_before=0,
        device="cpu",
        description_formatter=lambda metadata: metadata["text"],
    )
    monkeypatch.setattr(model, "clear_cuda_cache", lambda: None)
    return model


def test_pool_is_a_frozen_ordered_condition_with_shared_metadata():
    members = ["1", {"text": "29"}, "1"]
    pool = PooledCondition("combined", members, assay="RNA", cell_type="cell", metadata={"source": "test"})
    members.pop()
    assert pool.descriptions == ("1", {"text": "29"}, "1")
    with pytest.raises(FrozenInstanceError):
        pool.name = "different"
    assert pool.to_dict() == {
        "name": "combined",
        "descriptions": ["1", {"text": "29"}, "1"],
        "assay": "RNA",
        "cell_type": "cell",
        "tissue": None,
        "metadata": {"source": "test"},
    }


@pytest.mark.parametrize(
    ("descriptions", "error"), [([], ValueError), ("12", TypeError), ([2], TypeError), ([" "], ValueError)]
)
def test_pool_rejects_invalid_descriptions(descriptions, error):
    with pytest.raises(error):
        PooledCondition("invalid", descriptions)


def test_single_prediction_reduces_fp32_logits_and_retains_each_member(model):
    pool = PooledCondition("pool", ("1", {"text": "29"}, "1"))
    sequence = AnnotatedSequence("ACGT")
    expected = torch.stack([
        model.predict_sequence(sequence, condition=description, center=0).logits.float()
        for description in pool.descriptions
    ]).mean(dim=0)
    model.model.forward_calls.clear()

    prediction = model.predict_sequence(sequence, condition=pool, center=0, grouping="sequence")

    assert isinstance(prediction, ExpressionPrediction)
    assert prediction.condition is pool
    assert prediction.logits.dtype == torch.float32
    torch.testing.assert_close(prediction.logits, expected)
    assert prediction.expression == prediction.scalar() == pytest.approx(expected[0, 0, 0].item())
    assert prediction.track().values == pytest.approx(expected[0, 1:-1, 0].tolist())
    assert prediction.to_dict()["condition"] == pool.to_dict()
    member_tokens = prediction.description_tokens["members"]
    assert [token["description"] for token in member_tokens] == ["1", "29", "1"]
    assert [token["input_ids"].tolist() for token in member_tokens] == [[1], [2, 9], [1]]
    assert len(prediction.provenance["description_pool"]["members"]) == 3
    assert model.model.forward_calls[0][0] == (1, 3)


@pytest.mark.parametrize("grouping", ["serial", "no_grouping", "condition", "sequence", "auto"])
def test_mixed_predictions_keep_logical_order_and_remove_batch_padding(model, grouping):
    pool = PooledCondition("pool", ("1", "29", "1"))
    singleton = PooledCondition("singleton", ("3",))
    plain = Condition("plain", "4")
    short = AnnotatedSequence("AC", name="short")
    long = AnnotatedSequence("ACGT", name="long")
    expected = model.predict_sequence(short, condition=pool, center=0)

    predictions = model.predict_multiple_sequences(
        [short, long, short],
        conditions=[pool, plain, singleton],
        center=0,
        grouping=grouping,
        max_records_per_forward=2,
        show_progress=False,
    )

    assert [prediction.condition for prediction in predictions] == [pool, plain, singleton]
    assert [prediction.sequence.name for prediction in predictions] == ["short", "long", "short"]
    assert predictions[0].logits.shape == (1, 4, 1)
    torch.testing.assert_close(predictions[0].logits, expected.logits)
    assert predictions[1].logits.dtype == torch.bfloat16
    assert predictions[1].description_tokens["description"] == "4"
    assert predictions[1].provenance["record_index"] == 1
    independent = model.predict_sequence(short, condition="3", center=0)
    torch.testing.assert_close(predictions[2].logits, independent.logits.float())


def test_multi_sequence_prediction_broadcasts_a_whole_pool(model):
    pool = PooledCondition("pool", ("1", "2"))
    results = model.predict_multiple_sequences(
        ["AC", "GT"], condition=pool, center=0, show_progress=False, retention="scalars",
    )

    assert [result.condition for result in results] == [pool, pool]
    for result in results:
        assert result.expression == 16
        assert result.logits is None and result.description_tokens is None
        assert result.tokens is None and result.sequence is None


@pytest.mark.parametrize(("execution", "grouping"), [
    ("joint", "no_grouping"), ("joint", "condition"), ("separate", "sequence"),
])
def test_pooled_pairs_average_alleles_separately_and_keep_pair_members_together(model, execution, grouping):
    pool = PooledCondition("pool", ("1", "29", "1"))
    plain = Condition("plain", "3")
    pair = SequencePair(AnnotatedSequence("AC"), AnnotatedSequence("ACGT"))
    other = SequencePair(AnnotatedSequence("GC"), AnnotatedSequence("GT"))
    individual = [
        model.predict_pair(pair, condition=description, center=0)
        for description in pool.descriptions
    ]
    model.model.forward_calls.clear()
    limit = {"max_pairs_per_forward": 1} if execution == "joint" else {"max_records_per_forward": 2}
    predictions = model.predict_multiple_pairs(
        [pair, other],
        conditions=[pool, plain],
        center=0,
        pair_execution=execution,
        grouping=grouping,
        show_progress=False,
        **limit,
    )

    if execution == "joint":
        assert len(model.model.forward_calls) == 4
        for shape, rows, scores in model.model.forward_calls:
            assert shape == (1, 2)
            assert scores[0][0] == scores[0][1]
            assert rows[0] != rows[1]
    result, ordinary = predictions
    assert ordinary.condition is plain
    assert result.condition is result.ref.condition is result.alt.condition is pool
    for allele in ("ref", "alt"):
        expected = torch.stack([getattr(pred, allele).logits.float() for pred in individual]).mean(dim=0)
        torch.testing.assert_close(getattr(result, allele).logits, expected)
    assert result.delta_scalar() == pytest.approx(sum(pred.delta_scalar() for pred in individual) / 3)


def test_variant_interpreter_preserves_pool_and_alphagenome_rejects_it(model):
    pool = PooledCondition("pool", ("1", "29"))
    pair = SequencePair(AnnotatedSequence("AC"), AnnotatedSequence("AT"))
    scored = VariantInterpreter(model).score_sequence_pair(
        pair, condition=pool, scorer=ExpressionDeltaScorer(), center=0,
    )
    assert scored.condition is pool
    assert scored.condition_name == pool.name

    alphagenome = AlphaGenomeSequenceModel.__new__(AlphaGenomeSequenceModel)
    with pytest.raises(TypeError, match="AlphaGenomeSequenceModel does not support PooledCondition"):
        alphagenome.predict_sequence("AC", condition=pool)
    with pytest.raises(TypeError, match="multiple descriptions"):
        model.tokenize_description(pool)

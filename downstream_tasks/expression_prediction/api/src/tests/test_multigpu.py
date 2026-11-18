"""CUDA replica tests with real spawn IPC and an offline model fixture."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from time import sleep

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from gena_expression import AnnotatedSequence, PooledCondition, SequenceModel, SequencePair
from gena_expression.inference.workers import _CudaWorkerPool, _ModelLoadSpec, _cuda_devices


_SPEC = _ModelLoadSpec("unused::Model", Path("weights.pt"), Path("model.yaml"), "default")


def _cpu_replica(device, spec):
    assert not mp.current_process().daemon

    def forward(input_tensors):
        action = input_tensors.pop("action", None)
        if action == "raise":
            raise ValueError("replica test failure")
        if action == "exit":
            os._exit(20)
        sleep(input_tensors.pop("delay", 0))
        return input_tensors["values"] * 2

    return forward, {"model_module": spec.model_cls}


def _bad_start(device, spec):
    raise ValueError("replica setup failed")


def test_cuda_selector_validates_lists_and_preserves_legacy_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 6)
    assert _cuda_devices("cuda:0,2,5") == ("cuda:0", "cuda:2", "cuda:5")
    for legacy in (None, "cpu", "cuda", "cuda:0", torch.device("cuda", 0)):
        assert _cuda_devices(legacy) == ()
    for invalid in ("gpu:0,2", "gpu:0", "cuda:0,0", "cuda:0,6", "cuda:0,-2", "cuda:0,", "cpu,1"):
        with pytest.raises(ValueError, match="CUDA|multi-GPU"):
            _cuda_devices(invalid)


def test_invalid_selector_fails_before_loading_resources():
    with pytest.raises(ValueError, match="Use 'cuda'"):
        SequenceModel.load(
            "missing::model", "weights.pt", "model.yaml", "dna", "description",
            dna_max_seq_len=8, desc_max_seq_len=8, token_len_for_fetch=1, num_before=2,
            device="gpu:0,2",
        )


def test_spawned_cpu_pool_reuses_workers_and_bounds_lookahead():
    pool = _CudaWorkerPool(("first", "second"), _SPEC, initialize=_cpu_replica, timeout=30)
    submitted = []

    def jobs():
        for index in range(10):
            submitted.append(index)
            yield index, {"values": torch.tensor([index]), "delay": 0.5 if index == 0 else 0}

    try:
        results = pool.run(jobs())
        first_id, _, first_result = next(results)
        assert first_id == 0
        assert first_result.item() == 0
        assert submitted == list(range(4))
        tail = list(results)
        assert [group_id for group_id, _, _ in tail] == list(range(1, 10))
        assert [tensor.item() for _, _, tensor in tail] == list(range(2, 20, 2))
        assert {device for _, device, _ in tail} == {"first", "second"}
        assert list(pool.run([(0, {"values": torch.tensor([3])})]))[0][2].item() == 6
    finally:
        pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="workers are closed"):
        list(pool.run([]))


@pytest.mark.parametrize(
    ("action", "message"),
    [("raise", "forward batch 9.*replica test failure"), ("exit", "exited with code 20")],
)
def test_worker_failure_breaks_pool(action, message):
    pool = _CudaWorkerPool(("failing",), _SPEC, initialize=_cpu_replica, timeout=30)
    try:
        with pytest.raises(RuntimeError, match="(?s)" + message):
            list(pool.run([(9, {"action": action})]))
        with pytest.raises(RuntimeError, match="workers are closed"):
            list(pool.run([]))
    finally:
        pool.close()


def test_startup_error_identifies_replica():
    with pytest.raises(RuntimeError, match="(?s)CUDA worker failing failed during startup.*replica setup failed"):
        _CudaWorkerPool(("failing",), _SPEC, initialize=_bad_start, timeout=30)


@pytest.fixture
def load_options(tmp_path):
    model_file = tmp_path / "replica.py"
    model_file.write_text(
        """
from types import SimpleNamespace
import torch

class Replica(torch.nn.Module):
    def __init__(self, attn_implementation=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.attn_implementation = attn_implementation

    def forward(self, *, input_ids, attention_mask, desc_input_ids,
                desc_attention_mask, dataset_flag, labels, labels_mask, return_dict):
        if self.attn_implementation == "sdpa":
            assert torch.backends.cuda.math_sdp_enabled()
            assert not torch.backends.cuda.flash_sdp_enabled()
            assert not torch.is_autocast_enabled(input_ids.device.type)
        assert dataset_flag.numel() == input_ids.shape[0]
        description = (desc_input_ids * desc_attention_mask).sum(dim=-1).reshape(-1, 1)
        value = input_ids + description
        return SimpleNamespace(logits=value.unsqueeze(-1) * self.weight)
""",
        encoding="utf-8",
    )
    vocabulary = {token: index for index, token in enumerate(("[PAD]", "[CLS]", "[SEP]", "[UNK]", "A", "C", "G", "T"))}
    tokens = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokens.pre_tokenizer = Split("", behavior="isolated")
    tokens.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokens, cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    (tmp_path / "model.yaml").write_text("model_kwargs: {}\n", encoding="utf-8")
    torch.save({"weight": torch.ones(1)}, tmp_path / "weights.pt")
    return dict(
        model_cls=f"{model_file}::Replica", checkpoint=tmp_path / "weights.pt", config=tmp_path / "model.yaml",
        dna_tokenizer=tmp_path / "tokenizer", description_tokenizer=tmp_path / "tokenizer",
        dna_max_seq_len=16, desc_max_seq_len=8, token_len_for_fetch=1, num_before=2,
        description_formatter=lambda metadata: str(metadata["text"]),
        precision="float32",
    )


def _check_cuda_predictions(load_options, device, expected_devices):
    local = SequenceModel.load(device="cpu", **load_options)
    with SequenceModel.load(device=device, **load_options) as model:
        assert model.model is None
        assert model.provenance["devices"] == expected_devices
        sequences = [AnnotatedSequence("ACGT" + "AC" * index, name=f"row_{index}") for index in range(5)]
        pairs = [SequencePair(sequence, AnnotatedSequence("AG" + sequence.sequence[2:])) for sequence in sequences]
        common = dict(center=2, show_progress=False)
        requested = dict(
            conditions=[{"text": "C"}, "A", "C", {"text": "GAA"}, "A"],
            grouping="condition", max_records_per_forward=1, preprocessing_workers=1,
            prefetch_batches=2,
        )
        observed = model.predict_multiple_sequences(sequences, **common, **requested)
        expected = local.predict_multiple_sequences(sequences, **common, **requested)
        assert [p.sequence.name for p in observed] == [p.sequence.name for p in expected]
        assert [p.provenance["forward_group_id"] for p in observed] == [p.provenance["forward_group_id"] for p in expected]
        for actual, reference in zip(observed, expected, strict=True):
            assert actual.provenance["forward_device"] in expected_devices
            torch.testing.assert_close(actual.logits, reference.logits)
            assert not actual.logits.is_shared()
            assert actual.tokens.tokens == reference.tokens.tokens
            assert actual.description_tokens["description"] == reference.description_tokens["description"]

        pooled = PooledCondition("pool", ("A", {"text": "C"}, "GAA"))
        expected_logits = torch.stack([
            local.predict_sequence(sequences[0], center=2, condition=description).logits
            for description in pooled.descriptions
        ]).mean(dim=0)
        pooled_prediction = model.predict_sequence(sequences[0], center=2, condition=pooled)
        assert pooled_prediction.condition is pooled
        torch.testing.assert_close(pooled_prediction.logits, expected_logits)
        assert [
            member["description"] for member in pooled_prediction.description_tokens["members"]
        ] == ["A", "C", "GAA"]
        members = pooled_prediction.provenance["description_pool"]["members"]
        assert len(members) == 3
        assert {member["forward_device"] for member in members} <= set(expected_devices)
        expected_cap = 3 if len(expected_devices) == 1 else 2
        assert {member["max_records_per_forward"] for member in members} == {expected_cap}

        pair_options = dict(
            condition="C", pair_execution="joint", max_pairs_per_forward=2, preprocessing_backend="thread",
            preprocessing_workers=1,
        )
        observed_pairs = model.predict_multiple_pairs(pairs, **common, **pair_options)
        expected_pairs = local.predict_multiple_pairs(pairs, **common, **pair_options)
        for actual, reference in zip(observed_pairs, expected_pairs, strict=True):
            torch.testing.assert_close(actual.ref.logits, reference.ref.logits)
            torch.testing.assert_close(actual.alt.logits, reference.alt.logits)
            assert actual.ref.provenance["forward_group_id"] == actual.alt.provenance["forward_group_id"]
            assert actual.ref.provenance["dataset_flag_shape"][1] == 2

        from gena_expression.config import PredictionRetention, RetentionPolicy

        compact = RetentionPolicy(prediction=PredictionRetention(logits=False, tokens=False))
        result = model.predict_sequence(sequences[0], center=2, condition="C", retention=compact)
        assert result.logits is None
        assert result.tokens is None
        assert result.scalar() == local.predict_sequence(sequences[0], center=2, condition="C").scalar()
    with pytest.raises(RuntimeError, match="workers are closed"):
        model.predict_sequence(sequences[0], center=2, condition="C")


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires CUDA")
def test_spawned_real_cuda_worker(load_options, monkeypatch):
    monkeypatch.setattr("gena_expression.inference.model._cuda_devices", lambda device: ("cuda:0",) if device == "cuda:0" else ())
    _check_cuda_predictions(load_options, "cuda:0", ["cuda:0"])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_cuda_replicas_match_local_predictions(load_options):
    _check_cuda_predictions(load_options, "cuda:0,1", ["cuda:0", "cuda:1"])

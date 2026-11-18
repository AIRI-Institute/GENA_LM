"""A tiny checkpoint exercises the explicit SequenceModel.load contract."""

from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from gena_expression.inference import SequenceModel


def test_load_uses_explicit_tokenizers_and_lengths(tmp_path, monkeypatch):
    config = tmp_path / "model.yaml"
    config.write_text(
        "model_kwargs:\n  in_features: 2\n  out_features: 1\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "weights.tensors"
    weight = torch.tensor([[1.0, 2.0]])
    bias = torch.tensor([0.5])
    save_file({"weight": weight, "bias": bias}, str(checkpoint))
    tokenizer_calls = []

    def load_tokenizer(name_or_path, padding_side="right"):
        tokenizer_calls.append((name_or_path, padding_side))
        return SimpleNamespace(cls_token_id=1, sep_token_id=2, pad_token_id=0)

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load_tokenizer)
    model = SequenceModel.load(
        model_cls="torch.nn::Linear",
        checkpoint=checkpoint,
        config=config,
        dna_tokenizer="dna-tokenizer",
        description_tokenizer="description-tokenizer",
        dna_max_seq_len=64,
        desc_max_seq_len=32,
        token_len_for_fetch=3,
        num_before=8,
        device="cpu",
        description_formatter=lambda metadata: str(metadata),
    )

    assert tokenizer_calls == [
        ("dna-tokenizer", "right"),
        ("description-tokenizer", "left"),
    ]
    assert (model.dna_max_seq_len, model.desc_max_seq_len) == (64, 32)
    assert (model.token_len_for_fetch, model.num_before) == (3, 8)
    assert str(model.device) == "cpu"
    assert not model.model.training
    torch.testing.assert_close(model.model.weight, weight)
    torch.testing.assert_close(model.model.bias, bias)

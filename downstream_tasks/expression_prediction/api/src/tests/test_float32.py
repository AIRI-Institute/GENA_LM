"""Check explicit FP32 loading and the real SDPA forward on local weights."""

import json
import runpy
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import (
    ModernBertConfig,
    ModernBertModel,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3Model,
)

from gena_expression import AnnotatedSequence, SequenceModel, SequencePair


def _load_options(tmp_path, model_cls, model_kwargs, state_dict, device):
    tokens = ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "A", "C", "G", "T")
    vocabulary = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Split("", behavior="isolated")
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", unk_token="[UNK]"
    )
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    config = tmp_path / "model.yaml"
    config.write_text(json.dumps({"model_kwargs": model_kwargs}), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(state_dict, checkpoint)
    return {
        "model_cls": model_cls,
        "checkpoint": checkpoint,
        "config": config,
        "dna_tokenizer": tokenizer_path,
        "description_tokenizer": tokenizer_path,
        "dna_max_seq_len": 16,
        "desc_max_seq_len": 8,
        "token_len_for_fetch": 1,
        "num_before": 2,
        "device": device,
        "description_formatter": lambda metadata: str(metadata),
    }


def test_float32_casts_before_loading_checkpoint(tmp_path):
    module = tmp_path / "narrow.py"
    module.write_text(
        """
import torch

class Narrow(torch.nn.Module):
    def __init__(self, attn_implementation="legacy"):
        super().__init__()
        self.attn_implementation = attn_implementation
        self.weight = torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
        self.register_buffer("offset", torch.empty(1, dtype=torch.bfloat16))
        self.register_buffer("count", torch.empty(1, dtype=torch.long))
""",
        encoding="utf-8",
    )
    weights = {"weight": torch.tensor([1.0001]), "offset": torch.tensor([2.0001]), "count": torch.tensor([42])}
    options = _load_options(tmp_path, f"{module}::Narrow", {}, weights, "cpu")

    legacy = SequenceModel.load(**options)
    assert legacy.model.attn_implementation == "legacy"
    assert legacy.model.weight.dtype == torch.bfloat16

    predictor = SequenceModel.load(**options, precision="float32")
    assert predictor.model.attn_implementation == "sdpa"
    for name, value in weights.items():
        torch.testing.assert_close(getattr(predictor.model, name), value, rtol=0, atol=0)

    with pytest.raises(ValueError, match="precision"):
        SequenceModel.load(**options, precision="float16")


def _backends():
    return (
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
def test_expression_model_runs_fp32_sdpa_in_every_forward_layout(tmp_path, device):
    model_file = Path(__file__).resolve().parents[3] / "expression_model_final.py"
    expression_counts = runpy.run_path(str(model_file))["ExpressionCounts"]
    modern_path = tmp_path / "modern_dna_decoder"
    desc_path = tmp_path / "tiny_qwen3"
    dimensions = {
        "vocab_size": 16,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "max_position_embeddings": 32,
        "pad_token_id": 0,
    }
    ModernBertModel(ModernBertConfig(
        **dimensions, local_attention=8, cls_token_id=1, sep_token_id=2, reference_compile=False,
    )).save_pretrained(modern_path)
    Qwen3Model(Qwen3Config(**dimensions, num_key_value_heads=2, head_dim=8)).save_pretrained(desc_path)
    model_kwargs = {
        "config": None,
        "hf": True,
        "hf_model_name": str(modern_path),
        "hf_model_name_decoder": str(modern_path),
        "desc_model_name": str(desc_path),
        "dropout_prob": 0.0,
    }
    checkpoint_model = expression_counts(**model_kwargs, attn_implementation="sdpa")
    options = _load_options(
        tmp_path, f"{model_file}::ExpressionCounts", model_kwargs, checkpoint_model.state_dict(), device
    )
    predictor = SequenceModel.load(**options, precision="float32")
    components = [predictor.model.bert, predictor.model.desc_model, predictor.model.decoder]
    assert {component.config._attn_implementation for component in components} == {"sdpa"}
    assert {parameter.dtype for parameter in predictor.model.parameters()} == {torch.float32}

    layouts = []

    def verify_context(_, args, inputs):
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled(device)
        assert _backends() == (True, False, False, False)
        for key in ("input_ids", "attention_mask", "desc_input_ids", "desc_attention_mask"):
            assert inputs[key].dtype == torch.long
        assert inputs["dataset_flag"].dtype == torch.bool
        layouts.append(inputs["dataset_flag"].tolist())

    predictor.model.register_forward_pre_hook(verify_context, with_kwargs=True)
    api_options = dict(center=2, show_progress=False)
    before = _backends()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        ungrouped = predictor.predict_multiple_sequences(
            ["ACGT", "ACGTAC"], condition="A", **api_options,
        )
        condition_group = predictor.predict_multiple_sequences(
            ["ACGT", "ACGTAC"], condition="A", grouping="condition", **api_options,
        )
        sequence_group = predictor.predict_multiple_sequences(
            "ACGT", conditions=["A", "ACGT"], grouping="sequence", **api_options,
        )
        pair = SequencePair(AnnotatedSequence("ACGT"), AnnotatedSequence("AGGT"))
        joint = predictor.predict_pair(pair, condition="A", center=2, pair_execution="joint")
        assert torch.is_autocast_enabled(device)
    assert _backends() == before
    assert layouts == [
        [[False], [False]], [[False, False]], [[True, True]], [[False, False]],
    ]
    for prediction in [*ungrouped, *condition_group, *sequence_group, joint.ref, joint.alt]:
        assert prediction.logits.dtype == torch.float32
        assert torch.isfinite(prediction.logits).all()
    torch.testing.assert_close(ungrouped[0].logits, condition_group[0].logits)
    torch.testing.assert_close(ungrouped[1].logits, condition_group[1].logits)

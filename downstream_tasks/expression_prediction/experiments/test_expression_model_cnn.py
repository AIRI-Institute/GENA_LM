"""Wiring tests for :class:`ExpressionCountsCNN`.

``ExpressionCountsCNN.__init__`` downloads ModernGENA, a ModernBERT decoder and
Qwen3-Embedding, none of which is available on a bare CPU box. These tests build
the module tree directly and swap the three pretrained components for stubs, so
the parts written in this repo -- the CNN encoder, the projection, the CLS slot,
the fusion, the head and the loss -- are exercised end to end.

Runnable either under pytest or directly::

    python downstream_tasks/expression_prediction/test_expression_model_cnn.py
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from downstream_tasks.expression_prediction.alphagenome_cnn import AlphaGenomeStyleCNNEncoder
from downstream_tasks.expression_prediction.expression_model_cnn import ExpressionCountsCNN

HIDDEN = 32
DESC_HIDDEN = 24
STRIDE = 8  # 3 pooling stages


class _StubTower(nn.Module):
    """Stands in for ModernGENA / the ModernBERT decoder.

    Accepts ``inputs_embeds`` and multiplies by the attention mask so that a
    broken mask shows up in the output rather than being silently ignored.
    """

    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.config = SimpleNamespace(hidden_size=hidden)

    def forward(self, inputs_embeds=None, attention_mask=None, return_dict=True):
        hidden = self.proj(inputs_embeds)
        if attention_mask is not None:
            hidden = hidden * attention_mask[:, :, None].to(hidden.dtype)
        return SimpleNamespace(last_hidden_state=hidden, attentions=None)


class _StubDescModel(nn.Module):
    """Counts how many rows it was asked to encode, so de-duplication is testable."""

    def __init__(self, hidden):
        super().__init__()
        self.embed = nn.Embedding(64, hidden)
        self.config = SimpleNamespace(hidden_size=hidden)
        self.rows_seen = None

    def forward(self, input_ids=None, attention_mask=None, return_dict=True):
        self.rows_seen = input_ids.shape[0]
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


def build_stub_model(weight=0.5, num_pooling_stages=3, initial_channels=16):
    model = ExpressionCountsCNN.__new__(ExpressionCountsCNN)
    nn.Module.__init__(model)

    model.config = SimpleNamespace(use_return_dict=True, hidden_size=HIDDEN)
    model.gen_hidden_size = HIDDEN
    model.desc_hidden_size = DESC_HIDDEN

    model.bert = _StubTower(HIDDEN)
    model.decoder = _StubTower(HIDDEN)
    model.desc_model = _StubDescModel(DESC_HIDDEN)

    model.cnn = AlphaGenomeStyleCNNEncoder(
        initial_channels=initial_channels,
        channel_increment=8,
        num_pooling_stages=num_pooling_stages,
        gradient_checkpointing=False,
    )
    model.cnn_total_stride = model.cnn.total_stride

    model.cnn_proj = nn.Linear(model.cnn.output_channels, HIDDEN)
    model.cnn_ln = nn.LayerNorm(HIDDEN)
    model.cls_embedding = nn.Parameter(torch.randn(1, 1, HIDDEN) * 0.02)
    model.sep_embedding = nn.Parameter(torch.randn(1, 1, HIDDEN) * 0.02)
    model.dna_ln = nn.LayerNorm(HIDDEN)
    model.desc_ln = nn.LayerNorm(HIDDEN)
    model.desc_proj = nn.Linear(DESC_HIDDEN, HIDDEN)
    model.classifier = nn.Linear(HIDDEN, 1)

    model.activation = nn.Identity()
    model.loss_fct = nn.MSELoss(reduction="none")
    model.weight = weight
    model.use_deviation_loss = False
    model.use_multinomial_loss = False
    model.weight_deviation_loss = 1.0
    model.weight_multinomial_loss = 1.0
    return model


N_KEYS = 3


def make_batch(batch_size=2, window=64, desc_len=5, stride=STRIDE, n_keys=N_KEYS,
               distinct_desc=None):
    """One DNA window per gene, n_keys tracks of labels and descriptions."""
    seq_len = window // stride + 2
    if distinct_desc is None:
        distinct_desc = n_keys
    # Same track ids reused across genes, as in the real data.
    desc_index = torch.arange(n_keys).remainder(distinct_desc).repeat(batch_size, 1)
    return dict(
        dna_codes=torch.randint(0, 5, (batch_size, window)),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        labels=torch.randn(batch_size, n_keys, seq_len, 1),
        labels_mask=torch.zeros(batch_size, n_keys, seq_len, 1, dtype=torch.bool),
        desc_input_ids=torch.randint(0, 64, (batch_size, n_keys, desc_len)),
        desc_attention_mask=torch.ones(batch_size, n_keys, desc_len, dtype=torch.long),
        desc_index=desc_index,
    )


# ---------------------------------------------------------------- encode_dna --

def test_encode_dna_wraps_bins_in_cls_and_sep():
    model = build_stub_model()
    codes = torch.randint(0, 5, (3, 64))
    embeds = model.encode_dna(codes)
    assert embeds.shape == (3, 64 // STRIDE + 2, HIDDEN)
    # Position 0 is the shared learned CLS embedding, identical across the batch
    # and independent of the DNA content.
    assert torch.allclose(embeds[0, 0], embeds[1, 0])
    assert torch.allclose(embeds[:, 0], model.cls_embedding.expand(3, -1, -1)[:, 0])
    # ...and the last one is the shared learned SEP.
    assert torch.allclose(embeds[0, -1], embeds[1, -1])
    assert torch.allclose(embeds[:, -1], model.sep_embedding.expand(3, -1, -1)[:, 0])


def test_encode_dna_depends_on_sequence():
    model = build_stub_model()
    a = model.encode_dna(torch.zeros(1, 64, dtype=torch.long))
    b = model.encode_dna(torch.full((1, 64), 3, dtype=torch.long))
    assert not torch.allclose(a[:, 1:-1], b[:, 1:-1]), "bins ignore the input sequence"


# ------------------------------------------------------------------- shapes --

def test_forward_shapes():
    model = build_stub_model()
    for window in (32, 64, 256):
        batch = make_batch(window=window)
        out = model(**batch)
        assert out.logits.shape == (2 * N_KEYS, window // STRIDE + 2, 1), window


def test_forward_rejects_bad_shapes():
    model = build_stub_model()

    batch = make_batch()
    batch["dna_codes"] = batch["dna_codes"].unsqueeze(1)
    try:
        model(**batch)
    except ValueError as exc:
        assert "dna_codes must be (B, S)" in str(exc)
    else:
        raise AssertionError("expected ValueError for 3D dna_codes")

    batch = make_batch()
    batch["attention_mask"] = torch.ones(2, 999, dtype=torch.long)
    try:
        model(**batch)
    except ValueError as exc:
        assert "attention_mask must be" in str(exc)
    else:
        raise AssertionError("expected ValueError for a mismatched attention_mask")


def test_attention_mask_defaults_to_ones():
    model = build_stub_model()
    batch = make_batch()
    batch["attention_mask"] = None
    out = model(**batch)
    assert out.logits.shape == (2 * N_KEYS, 64 // STRIDE + 2, 1)


def test_attention_mask_is_actually_used():
    """Masking the bins must change the output, not be silently dropped."""
    model = build_stub_model()
    batch = make_batch()
    unmasked = model(**batch).logits

    masked_batch = dict(batch)
    mask = batch["attention_mask"].clone()
    mask[:, 4:] = 0
    masked_batch["attention_mask"] = mask
    masked = model(**masked_batch).logits

    assert not torch.allclose(unmasked, masked)


# ------------------------------------------------------- de-duplication ------

def test_dna_branch_runs_once_per_gene():
    """The tower must see B rows, not B * n_keys -- that is the whole point."""
    model = build_stub_model()
    batch = make_batch(batch_size=2, n_keys=N_KEYS)

    seen = {}
    inner = model.bert.forward

    def spy(inputs_embeds=None, **kw):
        seen["rows"] = inputs_embeds.shape[0]
        return inner(inputs_embeds=inputs_embeds, **kw)

    model.bert.forward = spy
    model(**batch)
    assert seen["rows"] == 2, f"tower ran on {seen['rows']} rows, expected 2 genes"


def test_description_encoder_runs_once_per_distinct_track():
    model = build_stub_model()
    # 2 genes x 3 tracks = 6 rows, but only 2 distinct descriptions.
    batch = make_batch(batch_size=2, n_keys=3, distinct_desc=2)
    model(**batch)
    assert model.desc_model.rows_seen == 2, (
        f"desc encoder ran on {model.desc_model.rows_seen} rows, expected 2 distinct"
    )


def test_description_dedup_scatters_the_right_embedding():
    """Rows sharing a description must get identical fused contributions."""
    model = build_stub_model()
    batch = make_batch(batch_size=1, n_keys=4, distinct_desc=2)
    # desc_index is [0, 1, 0, 1]; make the token ids agree with that grouping.
    ids = batch["desc_input_ids"]
    ids[0, 2] = ids[0, 0]
    ids[0, 3] = ids[0, 1]
    out = model(**batch).logits
    assert torch.allclose(out[0], out[2], atol=1e-5)
    assert torch.allclose(out[1], out[3], atol=1e-5)
    assert not torch.allclose(out[0], out[1])


def test_without_desc_index_every_row_is_encoded():
    model = build_stub_model()
    batch = make_batch(batch_size=2, n_keys=3)
    batch["desc_index"] = None
    model(**batch)
    assert model.desc_model.rows_seen == 6


def test_rejects_two_dimensional_desc_input():
    model = build_stub_model()
    batch = make_batch()
    batch["desc_input_ids"] = batch["desc_input_ids"][:, 0]
    try:
        model(**batch)
    except ValueError as exc:
        assert "desc_input_ids must be (B, N, D)" in str(exc)
    else:
        raise AssertionError("expected ValueError for 2D desc_input_ids")


# --------------------------------------------------------------------- loss --

def test_cls_only_loss():
    model = build_stub_model(weight=0.5)
    batch = make_batch()
    batch["labels_mask"][:, :, 0, 0] = True  # only the gene-level target

    out = model(**batch)
    assert out.cls_loss is not None
    assert out.other_loss is None
    assert torch.allclose(out.loss, out.cls_loss)


def test_cls_and_bin_loss_are_weighted():
    model = build_stub_model(weight=0.5)
    batch = make_batch()
    batch["labels_mask"][:] = True

    out = model(**batch)
    assert out.cls_loss is not None and out.other_loss is not None
    assert torch.allclose(out.loss, out.cls_loss + 0.5 * out.other_loss)


def test_no_loss_without_labels():
    model = build_stub_model()
    batch = make_batch()
    batch["labels"] = None
    batch["labels_mask"] = None
    out = model(**batch)
    assert out.loss is None
    assert out.logits is not None


def test_fully_masked_batch_yields_no_loss():
    model = build_stub_model()
    batch = make_batch()  # labels_mask is all False
    out = model(**batch)
    assert out.loss is None


def test_cls_loss_reads_position_zero():
    """The metric code indexes logits[:, 0, 0]; the loss must target the same slot."""
    model = build_stub_model()
    batch = make_batch()
    batch["labels_mask"][:, :, 0, 0] = True
    batch["labels"] = torch.zeros_like(batch["labels"])

    out = model(**batch)
    expected = out.logits[:, 0, 0].pow(2).mean()
    assert torch.allclose(out.cls_loss, expected, atol=1e-6)


# ----------------------------------------------------------------- backward --

def test_gradients_reach_every_trainable_component():
    model = build_stub_model()
    batch = make_batch()
    batch["labels_mask"][:] = True

    model(**batch).loss.backward()

    checked = {
        "cnn stem": model.cnn.embedder.stem.weight,
        "cnn block 0": model.cnn.blocks[0].block1.conv.weight,
        "cnn block 1": model.cnn.blocks[1].block1.conv.weight,
        "cnn block 2": model.cnn.blocks[2].block1.conv.weight,
        "cnn_proj": model.cnn_proj.weight,
        "cls_embedding": model.cls_embedding,
        "sep_embedding": model.sep_embedding,
        "tower": model.bert.proj.weight,
        "desc_proj": model.desc_proj.weight,
        "decoder": model.decoder.proj.weight,
        "classifier": model.classifier.weight,
    }
    for name, param in checked.items():
        assert param.grad is not None, f"{name} received no gradient"
        assert param.grad.abs().sum().item() > 0, f"{name} gradient is all zeros"


def test_cls_embedding_gets_gradient_from_cls_loss_alone():
    model = build_stub_model()
    batch = make_batch()
    batch["labels_mask"][:, :, 0, 0] = True  # gene-level target only

    model(**batch).loss.backward()
    assert model.cls_embedding.grad is not None
    assert model.cls_embedding.grad.abs().sum().item() > 0


def test_bf16_autocast_forward_backward():
    model = build_stub_model()
    batch = make_batch()
    batch["labels_mask"][:] = True

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(**batch)
    assert torch.isfinite(out.logits.float()).all()
    out.loss.float().backward()
    assert torch.isfinite(model.cnn.embedder.stem.weight.grad).all()


def test_return_tuple():
    model = build_stub_model()
    batch = make_batch()
    batch["labels_mask"][:] = True
    loss, logits = model(**batch, return_dict=False)
    assert loss.dim() == 0
    assert logits.shape == (2 * N_KEYS, 64 // STRIDE + 2, 1)


# ------------------------------------------------------------ freeze helper --

def test_freeze_token_embeddings():
    class _WithModernEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Module()
            self.embeddings.tok_embeddings = nn.Embedding(10, 4)

    class _WithBertEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Module()
            self.embeddings.word_embeddings = nn.Embedding(10, 4)

    modern = _WithModernEmbeddings()
    ExpressionCountsCNN._freeze_token_embeddings(modern, "test")
    assert not modern.embeddings.tok_embeddings.weight.requires_grad

    bert = _WithBertEmbeddings()
    ExpressionCountsCNN._freeze_token_embeddings(bert, "test")
    assert not bert.embeddings.word_embeddings.weight.requires_grad

    # Must not raise for a module without an embeddings attribute.
    ExpressionCountsCNN._freeze_token_embeddings(nn.Linear(2, 2), "test")


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)

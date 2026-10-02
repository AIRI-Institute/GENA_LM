"""Wiring tests for :class:`ExpressionCountsFlashzoi`.

``__init__`` loads Flashzoi, a ModernBERT decoder and Qwen3-Embedding, none of
which runs on a bare CPU box (flash-attn needs CUDA). These tests build the
module tree directly and swap all three for stubs, so the parts written in this
repo -- the projection, the CLS/SEP slots, the fusion, the head, the loss and the
n_keys broadcast -- are exercised end to end on CPU.

The real encoder is covered separately in ``test_flashzoi_encoder.py``.

Runnable either under pytest or directly::

    python downstream_tasks/expression_prediction/test_expression_model_flashzoi.py
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from downstream_tasks.expression_prediction.expression_model_flashzoi import (
    ExpressionCountsFlashzoi,
)

HIDDEN = 32
DESC_HIDDEN = 24
ENC_CHANNELS = 48
OUT_BINS = 6
SEQ_LEN = OUT_BINS + 2
N_KEYS = 3


class _StubEncoder(nn.Module):
    """Stands in for FlashzoiEncoder: (B, S) codes -> (B, out_bins, C)."""

    def __init__(self, out_bins=OUT_BINS, channels=ENC_CHANNELS, bin_size=32):
        super().__init__()
        self.embed = nn.Embedding(5, channels)
        self.out_bins = out_bins
        self.output_channels = channels
        self.bin_size = bin_size
        self.total_stride = bin_size
        self.rows_seen = None

    def forward(self, codes):
        self.rows_seen = codes.shape[0]
        # Pool the window down to out_bins so the output still depends on the input.
        x = self.embed(codes.long())                      # (B, S, C)
        x = nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.out_bins)
        return x.transpose(1, 2)                          # (B, out_bins, C)

    def parameter_report(self):
        return "stub encoder"


class _StubTower(nn.Module):
    """Stands in for ModernGENA / the ModernBERT decoder."""

    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.config = SimpleNamespace(
            hidden_size=hidden, use_return_dict=True, max_position_embeddings=1024
        )

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


def build_stub_model(weight=0.5, use_tower=False, out_bins=OUT_BINS):
    model = ExpressionCountsFlashzoi.__new__(ExpressionCountsFlashzoi)
    nn.Module.__init__(model)

    model.dna_encoder = _StubEncoder(out_bins=out_bins)
    model.out_bins = out_bins
    model.seq_len = out_bins + 2

    model.decoder = _StubTower(HIDDEN)
    model.use_tower = use_tower
    model.bert = _StubTower(HIDDEN) if use_tower else None
    model.desc_model = _StubDescModel(DESC_HIDDEN)

    model.config = SimpleNamespace(use_return_dict=True, hidden_size=HIDDEN)
    model.gen_hidden_size = HIDDEN
    model.desc_hidden_size = DESC_HIDDEN

    model.cnn_proj = nn.Linear(ENC_CHANNELS, HIDDEN)
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


def make_batch(batch_size=2, window=64, desc_len=5, n_keys=N_KEYS, seq_len=SEQ_LEN,
               distinct_desc=None):
    if distinct_desc is None:
        distinct_desc = n_keys
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
    embeds = model.encode_dna(torch.randint(0, 5, (3, 64)))
    assert embeds.shape == (3, SEQ_LEN, HIDDEN)
    assert torch.allclose(embeds[:, 0], model.cls_embedding.expand(3, -1, -1)[:, 0])
    assert torch.allclose(embeds[:, -1], model.sep_embedding.expand(3, -1, -1)[:, 0])


def test_encode_dna_length_is_out_bins_not_window_over_stride():
    """The crop is why this model cannot derive seq_len from the window size."""
    model = build_stub_model()
    for window in (64, 256, 1024):
        assert model.encode_dna(torch.randint(0, 5, (1, window))).shape[1] == SEQ_LEN


def test_encode_dna_depends_on_sequence():
    model = build_stub_model()
    a = model.encode_dna(torch.zeros(1, 64, dtype=torch.long))
    b = model.encode_dna(torch.full((1, 64), 3, dtype=torch.long))
    assert not torch.allclose(a[:, 1:-1], b[:, 1:-1]), "bins ignore the input sequence"


# ------------------------------------------------------------------- shapes --

def test_forward_shapes():
    for use_tower in (False, True):
        model = build_stub_model(use_tower=use_tower)
        out = model(**make_batch())
        assert out.logits.shape == (2 * N_KEYS, SEQ_LEN, 1), use_tower


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


def test_dataset_model_out_bins_mismatch_is_loud():
    """A dataset built with a different out_bins must fail fast, not silently."""
    model = build_stub_model(out_bins=OUT_BINS)
    batch = make_batch(seq_len=SEQ_LEN + 4)  # dataset thinks there are 4 more bins
    try:
        model(**batch)
    except (ValueError, RuntimeError) as exc:
        assert "attention_mask must be" in str(exc) or "shape" in str(exc).lower()
    else:
        raise AssertionError("a bin-count mismatch went through unnoticed")


def test_attention_mask_defaults_to_ones():
    model = build_stub_model()
    batch = make_batch()
    batch["attention_mask"] = None
    assert model(**batch).logits.shape == (2 * N_KEYS, SEQ_LEN, 1)


def test_attention_mask_is_actually_used():
    model = build_stub_model()
    batch = make_batch()
    unmasked = model(**batch).logits
    masked_batch = dict(batch)
    mask = batch["attention_mask"].clone()
    mask[:, 4:] = 0
    masked_batch["attention_mask"] = mask
    assert not torch.allclose(unmasked, model(**masked_batch).logits)


# ------------------------------------------------------- de-duplication ------

def test_dna_branch_runs_once_per_gene():
    """The encoder must see B rows, not B * n_keys -- that is the whole point."""
    model = build_stub_model()
    model(**make_batch(batch_size=2, n_keys=N_KEYS))
    assert model.dna_encoder.rows_seen == 2, (
        f"encoder ran on {model.dna_encoder.rows_seen} rows, expected 2 genes"
    )


def test_description_encoder_runs_once_per_distinct_track():
    model = build_stub_model()
    model(**make_batch(batch_size=2, n_keys=3, distinct_desc=2))
    assert model.desc_model.rows_seen == 2


def test_description_dedup_scatters_the_right_embedding():
    model = build_stub_model()
    batch = make_batch(batch_size=1, n_keys=4, distinct_desc=2)
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
    """tpm-only datasets: only position 0 is masked in, so loss == cls_loss."""
    model = build_stub_model(weight=0.5)
    batch = make_batch()
    batch["labels_mask"][:, :, 0, 0] = True
    out = model(**batch)
    assert out.cls_loss is not None and out.other_loss is None
    assert torch.allclose(out.loss, out.cls_loss)


def test_cls_and_bin_loss_are_weighted():
    model = build_stub_model(weight=0.5)
    batch = make_batch()
    batch["labels_mask"][:] = True
    out = model(**batch)
    assert torch.allclose(out.loss, out.cls_loss + 0.5 * out.other_loss)


def test_cls_loss_reads_position_zero():
    """The metric code indexes logits[:, 0, 0]; the loss must target the same slot."""
    model = build_stub_model()
    batch = make_batch()
    batch["labels_mask"][:, :, 0, 0] = True
    batch["labels"] = torch.zeros_like(batch["labels"])
    out = model(**batch)
    assert torch.allclose(out.cls_loss, out.logits[:, 0, 0].pow(2).mean(), atol=1e-6)


def test_no_loss_without_labels():
    model = build_stub_model()
    batch = make_batch()
    batch["labels"] = None
    batch["labels_mask"] = None
    out = model(**batch)
    assert out.loss is None and out.logits is not None


def test_fully_masked_batch_yields_no_loss():
    model = build_stub_model()
    assert model(**make_batch()).loss is None


# ----------------------------------------------------------------- backward --

def test_gradients_reach_every_trainable_component():
    model = build_stub_model()
    batch = make_batch()
    batch["labels_mask"][:] = True
    model(**batch).loss.backward()

    checked = {
        "encoder": model.dna_encoder.embed.weight,
        "cnn_proj": model.cnn_proj.weight,
        "cls_embedding": model.cls_embedding,
        "sep_embedding": model.sep_embedding,
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
    batch["labels_mask"][:, :, 0, 0] = True
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
    assert torch.isfinite(model.cnn_proj.weight.grad).all()


def test_return_tuple():
    model = build_stub_model()
    batch = make_batch()
    batch["labels_mask"][:] = True
    loss, logits = model(**batch, return_dict=False)
    assert loss.dim() == 0
    assert logits.shape == (2 * N_KEYS, SEQ_LEN, 1)


# ------------------------------------------------------------ freeze helper --

def test_cnn_alias_keeps_the_shared_runner_unmodified():
    """run_expression_finetuning_cnn.py line 385 does `model.cnn.parameters()`.

    Without this alias the runner crashes with AttributeError right after the
    model is built, and the only fix would be editing a file cnn_v1 also uses.
    """
    model = build_stub_model()
    assert model.cnn is model.dna_encoder
    # exactly what the runner does
    assert sum(p.numel() for p in model.cnn.parameters()) > 0


def test_freeze_token_embeddings():
    class _WithModernEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Module()
            self.embeddings.tok_embeddings = nn.Embedding(10, 4)

    modern = _WithModernEmbeddings()
    ExpressionCountsFlashzoi._freeze_token_embeddings(modern, "test")
    assert not modern.embeddings.tok_embeddings.weight.requires_grad
    ExpressionCountsFlashzoi._freeze_token_embeddings(nn.Linear(2, 2), "test")


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

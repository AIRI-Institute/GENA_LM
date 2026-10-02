"""Tests for :class:`FlashzoiEncoder` against the real pretrained weights.

Flashzoi's attention is ``flash_attn.modules.mha.MHA``, which needs a CUDA device
and half precision, so the forward tests are skipped without a GPU. The geometry
tests are pure arithmetic and always run.

Run from the GENA_LM root::

    CUDA_VISIBLE_DEVICES=6 python downstream_tasks/expression_prediction/test_flashzoi_encoder.py
"""

import torch
import torch.nn as nn

from downstream_tasks.expression_prediction.flashzoi_encoder import (
    LENGTH_MULTIPLE,
    FlashzoiEncoder,
)

import os

# A local directory on machines without internet (front3); override with
# FLASHZOI_MODEL=/path/to/flashzoi-replicate-0.
MODEL = os.environ.get("FLASHZOI_MODEL") or (
    f"{os.environ['GENALM_HOME']}/models/flashzoi-replicate-0"
    if os.environ.get("GENALM_HOME")
    and os.path.isdir(f"{os.environ['GENALM_HOME']}/models/flashzoi-replicate-0")
    else "johahi/flashzoi-replicate-0"
)
WINDOW = 32768
OUT_BINS = 1022
DIM = 1536

_HAS_CUDA = torch.cuda.is_available()
_CACHE = {}


def _encoder(**kwargs):
    """Loading the weights takes a few seconds; cache per kwarg signature."""
    key = tuple(sorted(kwargs.items()))
    if key not in _CACHE:
        enc = FlashzoiEncoder(MODEL, **kwargs).cuda().eval()
        _CACHE[key] = enc
    return _CACHE[key]


def _codes(batch=2, window=WINDOW):
    return torch.randint(0, 5, (batch, window), device="cuda")


def _bare_encoder(out_bins=OUT_BINS, pool=1, tap_bin_size=32):
    """Geometry-only instance: no weights, no CUDA."""
    enc = FlashzoiEncoder.__new__(FlashzoiEncoder)
    nn.Module.__init__(enc)
    enc.tap = "unet" if tap_bin_size == 32 else "transformer"
    enc.pool = pool
    enc.out_bins = out_bins
    enc.tap_bin_size = tap_bin_size
    enc.bin_size = tap_bin_size * pool
    enc.total_stride = enc.bin_size
    return enc


# ------------------------------------------------------------------ geometry --

def test_crop_offset_matches_the_dataset_formula():
    """The encoder and ExpressionDatasetCNN must derive the same shift.

    A mismatch here does not raise anywhere -- it silently slides the labels
    against the bins -- so it is checked explicitly.
    """
    cases = [
        # window, tap_bin_size, out_bins, pool
        (32768, 32, 1022, 1),
        (131072, 32, 1022, 1),
        (131072, 32, 1022, 4),
        (524288, 32, 1022, 1),
        (524288, 32, 1022, 4),
        (524288, 32, 1022, 8),
        (131072, 128, 1022, 1),   # transformer tap
    ]
    for window, tap_bin_size, out_bins, pool in cases:
        enc = _bare_encoder(out_bins=out_bins, pool=pool, tap_bin_size=tap_bin_size)
        enc.validate_length(window)
        # exactly what ExpressionDatasetCNN.__init__ computes, with its
        # cnn_total_stride = bin_size and n_bins = out_bins
        stride = tap_bin_size * pool
        ds_offset = ((window // stride - out_bins) // 2) * stride
        assert enc.crop_offset_bp(window) == ds_offset, (
            f"window={window} pool={pool}: encoder {enc.crop_offset_bp(window)} "
            f"!= dataset {ds_offset}"
        )


def test_rejects_windows_the_dataset_would_reject():
    """Both sides must agree on which geometries are legal.

    524288 with pool=6 gives a 192 bp grid that does not divide the window, so
    the two crops would disagree by 64 bp -- and nothing would raise at runtime.
    """
    for window, out_bins, pool, needle in [
        (524288, 1022, 6, "divisible by bin_size*pool"),
        (32768, 1023, 1, "symmetric"),
        (32768 + 1, 1022, 1, "multiple of"),
        (16384, 1022, 1, "at least"),
        # pool=2 would make the fine-grid difference even even though the
        # decoder-grid difference is odd: caught only on the pooled grid.
        (32768, 511, 2, "symmetric"),
    ]:
        enc = _bare_encoder(out_bins=out_bins, pool=pool)
        try:
            enc.validate_length(window)
        except ValueError as exc:
            assert needle in str(exc), f"{window}/{out_bins}/{pool}: {exc}"
        else:
            raise AssertionError(
                f"window={window} out_bins={out_bins} pool={pool} should have been rejected"
            )


def test_static_geometry_for_the_chosen_config():
    """32768 bp -> 1024 bins of 32 bp -> crop 1 bin per edge -> 1022."""
    enc = _bare_encoder()
    enc.validate_length(WINDOW)
    assert enc.crop_offset_bp(WINDOW) == 32
    assert enc.min_input_length == WINDOW
    assert enc.bin_size == 32
    assert enc.out_bins * enc.bin_size == 32704
    assert WINDOW % LENGTH_MULTIPLE == 0


def test_min_input_length_is_itself_valid():
    for pool, tap_bin_size in [(1, 32), (4, 32), (8, 32), (1, 128), (2, 128)]:
        enc = _bare_encoder(pool=pool, tap_bin_size=tap_bin_size)
        enc.validate_length(enc.min_input_length)  # must not raise


def test_center_crop_is_symmetric():
    x = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
    out = FlashzoiEncoder._center_crop(x, 6)
    assert out.shape == (1, 6, 1)
    assert out[0, 0, 0].item() == 2 and out[0, -1, 0].item() == 7
    assert torch.equal(FlashzoiEncoder._center_crop(x, 10), x)


# ------------------------------------------------------------------- forward --

def test_output_shape():
    enc = _encoder(out_bins=OUT_BINS, gradient_checkpointing=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc(_codes())
    assert out.shape == (2, OUT_BINS, DIM), out.shape
    assert torch.isfinite(out.float()).all()


def test_matches_upstream_get_embs_after_crop():
    """The rewritten trunk must reproduce borzoi_pytorch bit for bit.

    This is the test that catches a wrong skip connection or a swapped
    horizontal conv -- everything else would still produce the right shape.
    """
    from borzoi_pytorch import Borzoi

    enc = _encoder(
        out_bins=OUT_BINS,
        freeze_conv_tower=False,
        freeze_batchnorm=True,
        gradient_checkpointing=False,
    )
    ref = Borzoi.from_pretrained(MODEL, bins_to_return=OUT_BINS).cuda().eval()

    codes = _codes(batch=1)
    onehot = torch.nn.functional.embedding(codes.long(), enc.onehot_lut).transpose(1, 2)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mine = enc(codes)                                   # (1, bins, dim)
        theirs = ref.get_embs_after_crop(onehot).permute(0, 2, 1)

    assert mine.shape == theirs.shape, (mine.shape, theirs.shape)
    diff = (mine.float() - theirs.float()).abs().max().item()
    assert diff < 1e-2, f"max abs diff vs upstream: {diff}"
    del ref
    torch.cuda.empty_cache()


def test_transformer_tap_is_coarser_and_drops_the_upsampling_path():
    enc = _encoder(out_bins=OUT_BINS, tap="transformer", gradient_checkpointing=False)
    assert enc.bin_size == 128
    for name in ("upsampling_unet0", "separable0", "horizontal_conv0"):
        assert not hasattr(enc.borzoi, name), f"{name} should have been removed"
    # 131072 / 128 = 1024 transformer positions -> crop to 1022
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc(_codes(batch=1, window=131072))
    assert out.shape == (1, OUT_BINS, DIM), out.shape


def test_pooling_reduces_the_length():
    enc = _encoder(out_bins=OUT_BINS, pool=4, gradient_checkpointing=False)
    assert enc.bin_size == 128
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc(_codes(batch=1, window=131072))
    assert out.shape == (1, OUT_BINS, DIM), out.shape


def test_rejects_bad_input():
    enc = _encoder(out_bins=OUT_BINS, gradient_checkpointing=False)
    try:
        enc(torch.randint(0, 5, (2, 4, WINDOW), device="cuda"))
    except ValueError as exc:
        assert "(B, S)" in str(exc)
    else:
        raise AssertionError("expected ValueError for 3D codes")


# -------------------------------------------------------------------- freeze --

def test_dead_heads_are_removed():
    enc = _encoder(out_bins=OUT_BINS, gradient_checkpointing=False)
    for name in ("human_head", "mouse_head", "final_joined_convs", "final_softplus"):
        assert not hasattr(enc.borzoi, name), f"{name} would break DDP(find_unused=False)"
    assert all(p.requires_grad or True for p in enc.parameters())


def test_from_scratch_is_a_real_control():
    """pretrained=False must give the architecture WITHOUT the Borzoi weights.

    Also pins the init scheme we rely on: each FlashAttention keeps out_proj at
    zero, so every residual branch starts as the identity. If a future
    borzoi_pytorch lets post_init() overwrite that with xavier, an 8-layer
    transformer from scratch gets much harder to train -- and this test says so.
    """
    import torch.nn as nn

    scratch = FlashzoiEncoder(
        MODEL, out_bins=OUT_BINS, pretrained=False,
        freeze_conv_tower=False, freeze_batchnorm=False,
        gradient_checkpointing=False,
    ).cuda().eval()
    trained = _encoder(out_bins=OUT_BINS, gradient_checkpointing=False)

    # same topology
    assert scratch.output_channels == trained.output_channels
    assert (
        sum(p.numel() for p in scratch.borzoi.parameters())
        == sum(p.numel() for p in trained.borzoi.parameters())
    )
    # different weights
    a = scratch.borzoi.conv_dna.conv_layer.weight
    b = trained.borzoi.conv_dna.conv_layer.weight
    assert not torch.allclose(a, b), "from-scratch weights equal the pretrained ones"

    # residual branches start as the identity
    blk = scratch.borzoi.transformer[0][0].fn[1]
    assert (blk.mha.out_proj.weight == 0).all(), "out_proj not zero-initialised"

    # and it still runs
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = scratch(_codes(batch=1))
    assert out.shape == (1, OUT_BINS, DIM)
    assert torch.isfinite(out.float()).all()

    # every BatchNorm must be trainable and in train mode for a scratch run
    scratch.train()
    bns = [m for m in scratch.borzoi.modules() if isinstance(m, nn.BatchNorm1d)]
    assert all(m.training for m in bns), "BatchNorm frozen in a from-scratch run"
    del scratch
    torch.cuda.empty_cache()


def test_from_scratch_refuses_to_freeze():
    """Freezing random weights is never what anyone means."""
    for override in ({"freeze_conv_tower": True}, {"freeze_batchnorm": True}):
        kwargs = {"freeze_conv_tower": False, "freeze_batchnorm": False}
        kwargs.update(override)
        try:
            FlashzoiEncoder(MODEL, out_bins=OUT_BINS, pretrained=False, **kwargs)
        except ValueError as exc:
            assert "nothing worth freezing" in str(exc), str(exc)
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_batchnorm_stays_eval_after_train():
    """model.train() is called on every trainer step; BN must not follow."""
    enc = _encoder(out_bins=OUT_BINS, gradient_checkpointing=False)
    enc.train()
    bns = [m for m in enc.borzoi.modules() if isinstance(m, nn.BatchNorm1d)]
    assert bns, "no BatchNorm found -- did the architecture change?"
    assert all(not m.training for m in bns), "BatchNorm went back to train mode"
    enc.eval()


def test_frozen_tower_gets_no_gradients_and_bn_buffers_do_not_move():
    enc = FlashzoiEncoder(
        MODEL, out_bins=OUT_BINS, freeze_conv_tower=True, gradient_checkpointing=True
    ).cuda()
    enc.train()

    bn = next(m for m in enc.borzoi.res_tower.modules() if isinstance(m, nn.BatchNorm1d))
    before = bn.running_mean.clone()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc(_codes(batch=1))
    out.float().sum().backward()

    for name, p in enc.borzoi.conv_dna.named_parameters():
        assert p.grad is None, f"frozen conv_dna.{name} got a gradient"
    for name, p in enc.borzoi.res_tower.named_parameters():
        assert p.grad is None, f"frozen res_tower.{name} got a gradient"

    got_grad = [
        n for n, p in enc.borzoi.transformer.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert got_grad, "no transformer parameter received a gradient"
    assert torch.equal(bn.running_mean, before), (
        "BatchNorm running stats moved -- no_grad does not stop buffer updates"
    )
    del enc
    torch.cuda.empty_cache()


def test_gradient_checkpointing_gives_the_same_gradients():
    codes = _codes(batch=1)
    grads = {}
    for ckpt in (False, True):
        torch.manual_seed(0)
        enc = FlashzoiEncoder(
            MODEL, out_bins=OUT_BINS, freeze_conv_tower=True, gradient_checkpointing=ckpt
        ).cuda()
        enc.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = enc(codes)
        out.float().pow(2).mean().backward()
        p = enc.borzoi.separable0.conv_layer[1].weight
        grads[ckpt] = p.grad.detach().float().clone()
        del enc
        torch.cuda.empty_cache()
    diff = (grads[False] - grads[True]).abs().max().item()
    assert diff < 1e-3, f"checkpointing changed the gradient by {diff}"


if __name__ == "__main__":
    torch.manual_seed(0)
    needs_cuda = {
        "test_from_scratch_is_a_real_control",
        "test_from_scratch_refuses_to_freeze",
        "test_output_shape",
        "test_matches_upstream_get_embs_after_crop",
        "test_transformer_tap_is_coarser_and_drops_the_upsampling_path",
        "test_pooling_reduces_the_length",
        "test_rejects_bad_input",
        "test_dead_heads_are_removed",
        "test_batchnorm_stays_eval_after_train",
        "test_frozen_tower_gets_no_gradients_and_bn_buffers_do_not_move",
        "test_gradient_checkpointing_gives_the_same_gradients",
    }
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures, skipped = [], 0
    for name, fn in tests:
        if name in needs_cuda and not _HAS_CUDA:
            print(f"SKIP {name} (no CUDA)")
            skipped += 1
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures) - skipped}/{len(tests) - skipped} passed"
          f"{f', {skipped} skipped' if skipped else ''}")
    if failures:
        raise SystemExit(1)

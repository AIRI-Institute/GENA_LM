"""Tests for the AlphaGenome-style convolutional DNA encoder and window geometry.

Runnable either under pytest or directly::

    python downstream_tasks/expression_prediction/test_alphagenome_cnn.py

Only torch and numpy are required; nothing here touches pysam, pyBigWig or
transformers, so it runs on a bare CPU environment.
"""

import numpy as np
import torch

from downstream_tasks.expression_prediction.alphagenome_cnn import (
    PAD_CODE,
    AlphaGenomeStyleCNNEncoder,
    DownresBlock,
    RMSBatchNorm1d,
    StandardizedConv1d,
    bin_coords,
    build_channel_schedule,
    encode_nucleotides,
    reverse_complement_codes,
    window_for_tss,
)

SMALL = dict(
    initial_channels=16,
    channel_increment=8,
    num_pooling_stages=3,
    gradient_checkpointing=False,
)


# ---------------------------------------------------------------- encoding --

def test_encode_nucleotides():
    codes = encode_nucleotides("ACGTN")
    assert codes.tolist() == [0, 1, 2, 3, PAD_CODE]
    # IUPAC ambiguity codes and anything unexpected fall back to pad.
    assert encode_nucleotides("RYKMSW").tolist() == [PAD_CODE] * 6


def test_reverse_complement_codes():
    codes = encode_nucleotides("ACGTN")
    rc = reverse_complement_codes(codes)
    # revcomp("ACGTN") == "NACGT"
    assert rc.tolist() == encode_nucleotides("NACGT").tolist()
    # Involution.
    assert reverse_complement_codes(rc).tolist() == codes.tolist()


def test_onehot_pad_is_all_zero():
    encoder = AlphaGenomeStyleCNNEncoder(**SMALL)
    codes = torch.tensor([[0, 1, 2, 3, PAD_CODE]])
    onehot = encoder.one_hot(codes)
    assert onehot.shape == (1, 4, 5)
    assert torch.equal(onehot[0, :, :4], torch.eye(4))
    assert onehot[0, :, 4].abs().sum().item() == 0.0


# ---------------------------------------------------------- channel schedule --

def test_channel_schedule_matches_published_table():
    """768 / +128 / 7 stages must reproduce the AlphaGenome resolution table.

    Seven halvings take 1 bp to 128 bp, but channels grow only six times
    (768 + 6 * 128 = 1536), so the final stage keeps its channel count.
    """
    schedule = build_channel_schedule(768, 128, 7)
    assert schedule == [768, 896, 1024, 1152, 1280, 1408, 1536, 1536]


def test_channel_schedule_explicit_increments():
    assert build_channel_schedule(256, 128, 3, num_channel_increments=3) == [256, 384, 512, 640]
    assert build_channel_schedule(256, 128, 3, num_channel_increments=0) == [256, 256, 256, 256]


# ------------------------------------------------------------------- shapes --

def test_forward_shapes_multiple_lengths():
    encoder = AlphaGenomeStyleCNNEncoder(**SMALL)
    assert encoder.total_stride == 8
    # 3 pooling stages but 2 channel increments by default: 16 -> 24 -> 32 -> 32.
    assert encoder.channels == [16, 24, 32, 32]
    assert encoder.output_channels == 32

    for length in (64, 256, 1024):
        codes = torch.randint(0, 5, (2, length))
        out = encoder(codes)
        assert out.shape == (2, length // 8, 32), (length, out.shape)
        assert encoder.output_length(length) == length // 8


def test_forward_rejects_indivisible_length():
    encoder = AlphaGenomeStyleCNNEncoder(**SMALL)
    codes = torch.randint(0, 5, (1, 100))  # 100 % 8 != 0
    try:
        encoder(codes)
    except ValueError as exc:
        assert "not divisible by 8" in str(exc), str(exc)
    else:
        raise AssertionError("expected ValueError for indivisible sequence length")


def test_exact_configuration_builds():
    """The published configuration must be constructible and shape-correct."""
    encoder = AlphaGenomeStyleCNNEncoder(
        initial_channels=768,
        channel_increment=128,
        num_pooling_stages=7,
        gradient_checkpointing=False,
    )
    assert encoder.output_channels == 1536
    assert encoder.total_stride == 128
    out = encoder(torch.randint(0, 5, (1, 256)))
    assert out.shape == (1, 2, 1536)


# ----------------------------------------------------------------- backward --

def _grad_reaches_every_stage(encoder, length=128):
    codes = torch.randint(0, 5, (2, length))
    encoder(codes).pow(2).mean().backward()

    stem_grad = encoder.embedder.stem.weight.grad
    assert stem_grad is not None and stem_grad.abs().sum() > 0, "no gradient at the stem"

    for i, block in enumerate(encoder.blocks):
        for name, param in block.named_parameters():
            assert param.grad is not None, f"block {i}: {name} has no gradient"
        total = sum(p.grad.abs().sum() for p in block.parameters())
        assert total > 0, f"block {i}: all gradients are zero"


def test_backward_reaches_stem_and_every_block():
    _grad_reaches_every_stage(AlphaGenomeStyleCNNEncoder(**SMALL))


def test_backward_with_gradient_checkpointing():
    cfg = dict(SMALL)
    cfg["gradient_checkpointing"] = True
    encoder = AlphaGenomeStyleCNNEncoder(**cfg)
    encoder.train()
    _grad_reaches_every_stage(encoder)


def test_checkpointing_matches_plain_forward():
    torch.manual_seed(0)
    plain = AlphaGenomeStyleCNNEncoder(**SMALL)
    cfg = dict(SMALL)
    cfg["gradient_checkpointing"] = True
    checkpointed = AlphaGenomeStyleCNNEncoder(**cfg)
    checkpointed.load_state_dict(plain.state_dict())

    plain.eval()
    checkpointed.eval()
    codes = torch.randint(0, 5, (2, 128))
    with torch.no_grad():
        assert torch.allclose(plain(codes), checkpointed(codes), atol=1e-5)


# ------------------------------------------------------------ RMSBatchNorm --

def test_rms_batch_norm_does_not_subtract_mean():
    """A plain BatchNorm would centre this input; RMS normalisation must not."""
    norm = RMSBatchNorm1d(3)
    norm.train()
    x = torch.full((4, 3, 16), 5.0)
    out = norm(x)
    # x / rms(x) == 1 for a constant input, so the mean stays far from zero.
    assert out.mean().item() > 0.9, out.mean().item()


def test_rms_batch_norm_ema_and_eval():
    norm = RMSBatchNorm1d(2, decay=0.9)
    assert torch.allclose(norm.running_mean_square, torch.ones(2))

    norm.train()
    x = torch.full((2, 2, 8), 3.0)  # mean square == 9
    norm(x)
    expected = 0.9 * 1.0 + 0.1 * 9.0
    assert torch.allclose(norm.running_mean_square, torch.full((2,), expected), atol=1e-5)

    # In eval mode the running statistic is used, not the batch statistic.
    norm.eval()
    with torch.no_grad():
        out = norm(torch.full((1, 2, 4), 3.0))
    assert torch.allclose(out, torch.full((1, 2, 4), 3.0 / expected**0.5), atol=1e-4)


def test_rms_batch_norm_rejects_wrong_shape():
    norm = RMSBatchNorm1d(3)
    for bad in (torch.zeros(2, 3), torch.zeros(2, 4, 8)):
        try:
            norm(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for shape {tuple(bad.shape)}")


# ------------------------------------------------------- StandardizedConv1d --

def test_standardized_conv_weights_are_standardized():
    conv = StandardizedConv1d(8, 16, kernel_size=5, padding=2)
    with torch.no_grad():
        conv.weight.normal_(mean=3.0, std=2.0)  # deliberately off-centre
    w = conv.standardized_weight()
    per_filter_mean = w.mean(dim=(1, 2))
    assert per_filter_mean.abs().max().item() < 1e-5, per_filter_mean.abs().max().item()
    # Variance is rescaled by 1/fan_in, so the weight norm is ~1 per filter.
    per_filter_norm = w.flatten(1).norm(dim=1)
    assert torch.allclose(per_filter_norm, torch.ones(16), atol=1e-4), per_filter_norm


def test_standardized_convolution_can_be_disabled():
    cfg = dict(SMALL)
    cfg["use_standardized_convolution"] = False
    encoder = AlphaGenomeStyleCNNEncoder(**cfg)
    assert not any(isinstance(m, StandardizedConv1d) for m in encoder.modules())
    assert encoder(torch.randint(0, 5, (1, 64))).shape == (1, 8, 32)


# -------------------------------------------------------------- DownresBlock --

def test_downres_block_zero_pads_the_residual():
    """With both conv branches zeroed, the output must be the padded residual."""
    block = DownresBlock(4, 6, kernel_size=5, normalization="none", standardized=False)
    with torch.no_grad():
        for conv in (block.block1.conv, block.block2.conv):
            conv.weight.zero_()
            conv.bias.zero_()

    x = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)
    out = block(x)
    assert out.shape == (1, 6, 4)

    expected = torch.nn.functional.max_pool1d(
        torch.nn.functional.pad(x, (0, 0, 0, 2)), kernel_size=2, stride=2
    )
    assert torch.allclose(out, expected), (out, expected)
    # The two appended channels carry zeros, not copies of real channels.
    assert out[0, 4:].abs().sum().item() == 0.0


def test_downres_block_refuses_to_shrink_channels():
    try:
        DownresBlock(16, 8)
    except ValueError as exc:
        assert "cannot shrink channels" in str(exc)
    else:
        raise AssertionError("expected ValueError when out_channels < in_channels")


# --------------------------------------------------------- mixed precision --

def test_bf16_autocast_forward_backward():
    encoder = AlphaGenomeStyleCNNEncoder(**SMALL)
    codes = torch.randint(0, 5, (2, 128))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = encoder(codes)
    assert torch.isfinite(out.float()).all(), "non-finite activations under bf16 autocast"
    out.float().pow(2).mean().backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"{name} has no gradient under autocast"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


# ------------------------------------------------------------ window layout --

def test_window_is_centred_on_tss():
    start, end = window_for_tss(tss=500_000, chrom_len=10_000_000, window_len=131_072, strand="+")
    assert end - start == 131_072
    assert start == 500_000 - 65_536
    assert start <= 500_000 < end


def test_window_slides_inside_chromosome_ends():
    # Near the start: shifted right, never negative.
    start, end = window_for_tss(tss=10, chrom_len=1_000_000, window_len=1024, strand="+")
    assert (start, end) == (0, 1024)

    # Near the end: shifted left, never past the contig.
    start, end = window_for_tss(tss=999_990, chrom_len=1_000_000, window_len=1024, strand="+")
    assert (start, end) == (1_000_000 - 1024, 1_000_000)

    # Contig shorter than the window: padded on the right.
    start, end = window_for_tss(tss=100, chrom_len=500, window_len=1024, strand="+")
    assert (start, end) == (0, 1024)
    assert end > 500  # the tail is padding


def test_window_offset_follows_transcription_direction():
    forward = window_for_tss(1_000_000, 10_000_000, 1024, "+", offset=256)
    reverse = window_for_tss(1_000_000, 10_000_000, 1024, "-", offset=256)
    assert forward[0] == 1_000_000 + 256 - 512
    assert reverse[0] == 1_000_000 - 256 - 512


def test_bin_coords_forward_strand():
    starts, ends = bin_coords(1000, 1064, "+", stride=8, n_bins=8)
    assert starts.tolist() == [1000, 1008, 1016, 1024, 1032, 1040, 1048, 1056]
    assert ends.tolist() == [s + 8 for s in starts.tolist()]
    # process_region_signals reads starts[0] / ends[-1] on the forward strand.
    assert starts[0] == 1000 and ends[-1] == 1064


def test_bin_coords_reverse_strand():
    starts, ends = bin_coords(1000, 1064, "-", stride=8, n_bins=8)
    # Bin 0 is the most upstream position in transcription order, i.e. the
    # rightmost genomic bin.
    assert ends[0] == 1064 and starts[0] == 1056
    # process_region_signals reads starts[-1] / ends[0] on the reverse strand.
    assert starts[-1] == 1000 and ends[0] == 1064


def test_bins_tile_the_window_exactly():
    for strand in ("+", "-"):
        starts, ends = bin_coords(0, 128, strand, stride=8, n_bins=16)
        covered = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
        assert np.array_equal(np.sort(covered), np.arange(128)), strand


def test_bin_order_matches_reverse_complemented_sequence():
    """Bin i of the model input must correspond to bin i of the genomic layout.

    The dataset reverse-complements the window for '-' genes, so array position
    i covers genomic coordinates counted from the right-hand end.
    """
    window = "AAAACCCCGGGGTTTT"  # 16 bp, stride 4 -> 4 bins
    start, end, stride, n_bins = 0, 16, 4, 4

    codes = encode_nucleotides(window)
    rc_codes = reverse_complement_codes(codes)
    starts, ends = bin_coords(start, end, "-", stride=stride, n_bins=n_bins)

    for i in range(n_bins):
        model_bin = rc_codes[i * stride : (i + 1) * stride]
        genomic_bin = reverse_complement_codes(codes[starts[i] : ends[i]])
        assert model_bin.tolist() == genomic_bin.tolist(), i


# ------------------------------------------------------------------ dropout --

def test_dropout_is_off_by_default():
    """Default config must be bit-identical to the version without dropout."""
    encoder = AlphaGenomeStyleCNNEncoder(**SMALL)
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d))
                   for m in encoder.modules())
    encoder.train()
    codes = torch.randint(0, 5, (2, 128))
    torch.manual_seed(0); a = encoder(codes)
    torch.manual_seed(1); b = encoder(codes)
    assert torch.allclose(a, b), "default encoder must be deterministic in train mode"


def test_dropout_perturbs_only_in_train_mode():
    cfg = dict(SMALL); cfg["dropout"] = 0.5
    encoder = AlphaGenomeStyleCNNEncoder(**cfg)
    codes = torch.randint(0, 5, (2, 128))

    encoder.train()
    torch.manual_seed(0); a = encoder(codes)
    torch.manual_seed(1); b = encoder(codes)
    assert not torch.allclose(a, b), "dropout must randomise the train-mode output"

    encoder.eval()
    with torch.no_grad():
        assert torch.allclose(encoder(codes), encoder(codes)), "eval must be deterministic"


def test_dropout_is_channel_wise_by_default():
    """Dropout1d zeroes whole feature maps; neighbouring bp are too correlated
    for element-wise dropout to remove much information."""
    cfg = dict(SMALL); cfg["dropout"] = 0.5
    enc_channels = AlphaGenomeStyleCNNEncoder(**cfg)
    assert any(isinstance(m, torch.nn.Dropout1d) for m in enc_channels.modules())

    cfg["dropout_channels"] = False
    enc_elements = AlphaGenomeStyleCNNEncoder(**cfg)
    assert any(isinstance(m, torch.nn.Dropout) and not isinstance(m, torch.nn.Dropout1d)
               for m in enc_elements.modules())


def test_dropout_placed_between_activation_and_conv():
    cfg = dict(SMALL); cfg["dropout"] = 0.3
    encoder = AlphaGenomeStyleCNNEncoder(**cfg)
    block = encoder.blocks[0].block1
    assert isinstance(block.drop, torch.nn.Dropout1d)
    # order in forward: norm -> gelu -> drop -> conv
    assert hasattr(block, "norm") and hasattr(block, "conv")


def test_dropout_backward_still_reaches_every_stage():
    cfg = dict(SMALL); cfg["dropout"] = 0.2
    encoder = AlphaGenomeStyleCNNEncoder(**cfg)
    encoder.train()
    _grad_reaches_every_stage(encoder)


# ------------------------------------------------------------------ report --

def test_parameter_report_runs():
    encoder = AlphaGenomeStyleCNNEncoder(**SMALL)
    report = encoder.parameter_report()
    assert "total:" in report
    assert report.count("block") == encoder.num_pooling_stages


def _report_configurations():
    print("\n--- parameter counts ---")
    for name, kwargs in (
        ("scaled  (256, +128, 7 stages)", dict(initial_channels=256)),
        ("exact   (768, +128, 7 stages)", dict(initial_channels=768)),
    ):
        encoder = AlphaGenomeStyleCNNEncoder(
            channel_increment=128, num_pooling_stages=7, gradient_checkpointing=False, **kwargs
        )
        total = sum(p.numel() for p in encoder.parameters())
        print(f"{name}: {total:,} params, out {encoder.output_channels} ch @ "
              f"{encoder.total_stride} bp/bin")

        window = 131_072
        elements = sum(
            (window // (2**i)) * c for i, c in enumerate(encoder.channels)
        )
        print(f"    activations @ {window} bp, bf16, 1 tensor/stage: "
              f"{2 * elements / 1024**2:.0f} MB/sample")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
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
    _report_configurations()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)

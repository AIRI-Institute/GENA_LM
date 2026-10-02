"""AlphaGenome-inspired convolutional DNA encoder.

Reusable building blocks that turn raw nucleotides into a downsampled
representation suitable for a long-range transformer tower:

    dna codes -> one-hot -> DNAEmbedder -> DownresBlock x N -> (B, S / 2**N, C)

The structure follows the AlphaGenome encoder (local convolutional stem,
residual conv blocks, +128 channels per stage, MaxPool downsampling), but the
default configuration is deliberately scaled down. Every published knob is
exposed so the exact configuration can be reproduced; see
``AlphaGenomeStyleCNNEncoder`` for the list of deviations.

Nucleotide codes used throughout: A=0, C=1, G=2, T=3, N/pad=4. ``N`` maps to an
all-zero one-hot row, so padded positions contribute nothing to the stem.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NUC_A, NUC_C, NUC_G, NUC_T, NUC_N = 0, 1, 2, 3, 4
NUC_VOCAB_SIZE = 5
PAD_CODE = NUC_N

# ASCII -> nucleotide code. Anything that is not ACGT (including N and IUPAC
# ambiguity codes) becomes PAD_CODE, which one-hot encodes to an all-zero row.
_CODE_TABLE = np.full(256, PAD_CODE, dtype=np.uint8)
for _char, _code in zip(b"ACGT", (NUC_A, NUC_C, NUC_G, NUC_T)):
    _CODE_TABLE[_char] = _code


def build_onehot_lookup() -> torch.Tensor:
    """(5, 4) lookup table; ACGT -> identity rows, N -> all zeros."""
    lut = torch.zeros(NUC_VOCAB_SIZE, 4, dtype=torch.float32)
    lut[NUC_A, 0] = 1.0
    lut[NUC_C, 1] = 1.0
    lut[NUC_G, 2] = 1.0
    lut[NUC_T, 3] = 1.0
    return lut


def encode_nucleotides(sequence: str) -> np.ndarray:
    """Uppercase ACGTN string -> uint8 codes."""
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    return _CODE_TABLE[raw]


def reverse_complement_codes(codes: np.ndarray) -> np.ndarray:
    """Reverse-complement in code space: A<->T, C<->G, pad stays pad."""
    complemented = np.where(codes < 4, 3 - codes, PAD_CODE).astype(np.uint8)
    return complemented[::-1]


def window_for_tss(
    tss: int,
    chrom_len: int,
    window_len: int,
    strand: str,
    offset: int = 0,
) -> Tuple[int, int]:
    """Genomic window ``[start, end)`` of exactly ``window_len`` bp around a TSS.

    The window is centred on the TSS (plus ``offset``, measured downstream along
    transcription). If that would run off a chromosome end the window is slid
    back inside, so only contigs shorter than ``window_len`` end up padded --
    relevant for fragmented multispecies assemblies.
    """
    signed_offset = offset if strand == "+" else -offset
    start = tss + signed_offset - window_len // 2
    end = start + window_len

    if chrom_len >= window_len:
        if start < 0:
            start, end = 0, window_len
        elif end > chrom_len:
            start, end = chrom_len - window_len, chrom_len
    else:
        start, end = 0, window_len  # short contig -> right-padded
    return int(start), int(end)


def bin_coords(
    start: int,
    end: int,
    strand: str,
    stride: int,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genomic ``[start, end)`` of every bin, ordered along transcription.

    The layout matches ``ExpressionDataset.process_region_signals``' expectations:
    for '+' the starts ascend so ``starts[0]``/``ends[-1]`` bound the region, for
    '-' they descend so ``starts[-1]``/``ends[0]`` do.
    """
    if strand == "+":
        starts = start + np.arange(n_bins, dtype=np.int64) * stride
        ends = starts + stride
    else:
        ends = end - np.arange(n_bins, dtype=np.int64) * stride
        starts = ends - stride
    return starts, ends


def build_channel_schedule(
    initial_channels: int,
    channel_increment: int,
    num_pooling_stages: int,
    num_channel_increments: Optional[int] = None,
) -> List[int]:
    """Channel count at each resolution, from 1 bp down to 2**num_pooling_stages bp.

    Returns ``num_pooling_stages + 1`` entries. ``num_channel_increments``
    defaults to ``num_pooling_stages - 1``, which reproduces the published
    AlphaGenome table exactly: with (768, 128, 7) it yields
    768, 896, 1024, 1152, 1280, 1408, 1536, 1536 -- the last pooling stage
    keeps the channel count, matching "768 + 6 x 128 = 1536" alongside seven
    halvings of the sequence length.
    """
    if num_channel_increments is None:
        num_channel_increments = max(num_pooling_stages - 1, 0)
    if num_channel_increments > num_pooling_stages:
        raise ValueError(
            f"num_channel_increments ({num_channel_increments}) cannot exceed "
            f"num_pooling_stages ({num_pooling_stages})"
        )

    channels = [int(initial_channels)]
    for stage in range(num_pooling_stages):
        grow = channel_increment if stage < num_channel_increments else 0
        channels.append(channels[-1] + int(grow))
    return channels


class RMSBatchNorm1d(nn.Module):
    """BatchNorm-like normalisation that divides by the RMS without centering.

    For each channel, statistics are taken over the batch and sequence axes::

        rms = sqrt(mean(x ** 2) + eps)
        out = gamma * (x / rms) + beta

    An exponential moving average of the per-channel mean square is kept
    (decay 0.9 by default) and used at inference time, mirroring AlphaGenome.

    Notes
    -----
    * Statistics are accumulated in fp32 even under autocast, so the module is
      safe in bf16/fp16 training.
    * Under DDP the per-rank statistics are averaged across ranks
      (``sync_ddp=True``); without this, replicas would drift apart because the
      EMA buffer is not itself gradient-synchronised.
    * Padded positions are *not* excluded from the statistics. AlphaGenome does
      not mask them either, and with the window policy used by the dataset
      padding is rare (only for contigs shorter than the window).
    """

    def __init__(
        self,
        num_channels: int,
        decay: float = 0.9,
        eps: float = 1e-5,
        sync_ddp: bool = True,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.decay = float(decay)
        self.eps = float(eps)
        self.sync_ddp = bool(sync_ddp)

        self.gamma = nn.Parameter(torch.ones(self.num_channels))
        self.beta = nn.Parameter(torch.zeros(self.num_channels))
        self.register_buffer("running_mean_square", torch.ones(self.num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S)
        if x.dim() != 3:
            raise ValueError(f"RMSBatchNorm1d expects (B, C, S), got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"RMSBatchNorm1d configured for {self.num_channels} channels, "
                f"got {x.shape[1]}"
            )

        if self.training:
            mean_square = x.detach().float().pow(2).mean(dim=(0, 2))
            if self.sync_ddp and dist.is_available() and dist.is_initialized():
                dist.all_reduce(mean_square, op=dist.ReduceOp.SUM)
                mean_square = mean_square / dist.get_world_size()
            self.running_mean_square.mul_(self.decay).add_(
                mean_square, alpha=1.0 - self.decay
            )
            # Keep the graph on the live statistics, as BatchNorm does.
            stat = x.float().pow(2).mean(dim=(0, 2))
        else:
            stat = self.running_mean_square.float()

        inv_rms = torch.rsqrt(stat + self.eps).to(x.dtype)
        gamma = self.gamma.to(x.dtype)
        beta = self.beta.to(x.dtype)
        return x * (gamma * inv_rms).view(1, -1, 1) + beta.view(1, -1, 1)

    def extra_repr(self) -> str:
        return f"{self.num_channels}, decay={self.decay}, eps={self.eps}"


class StandardizedConv1d(nn.Conv1d):
    """Conv1d with scaled weight standardization (NFNet-style).

    Weights are standardised over the (in_channels, kernel) axes and rescaled
    by ``1 / sqrt(fan_in)`` times a learned per-filter gain. This decouples the
    effective learning rate from the weight scale and is what AlphaGenome uses
    in place of plain convolutions.

    Setting ``use_standardized_convolution=False`` in the encoder swaps this for
    a plain ``nn.Conv1d``. The plain version trains fine but tends to need more
    careful LR tuning in deep residual conv stacks, because nothing keeps the
    per-filter weight norm from drifting.
    """

    def __init__(self, *args, eps: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = float(eps)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1))

    def standardized_weight(self) -> torch.Tensor:
        weight = self.weight.float()
        mean = weight.mean(dim=(1, 2), keepdim=True)
        var = weight.var(dim=(1, 2), keepdim=True, unbiased=False)
        fan_in = weight[0].numel()
        scale = torch.rsqrt(torch.clamp(var * fan_in, min=self.eps))
        return ((weight - mean) * scale * self.gain.float()).to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(x, self.standardized_weight(), self.bias)


def make_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    standardized: bool,
    bias: bool = True,
) -> nn.Conv1d:
    """'same' padding convolution; kernel sizes are required to be odd."""
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd for 'same' padding, got {kernel_size}")
    cls = StandardizedConv1d if standardized else nn.Conv1d
    return cls(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


def make_norm(kind: str, num_channels: int, rms_decay: float) -> nn.Module:
    kind = (kind or "rms_batch_norm").lower()
    if kind == "rms_batch_norm":
        return RMSBatchNorm1d(num_channels, decay=rms_decay)
    if kind == "batch_norm":
        return nn.BatchNorm1d(num_channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(
        f"Unknown normalization {kind!r}; expected 'rms_batch_norm', 'batch_norm' or 'none'"
    )


class ConvBlock(nn.Module):
    """norm -> GELU -> dropout -> conv(kernel_size, 'same').

    Dropout sits between the activation and the convolution, the usual place for
    a pre-activation residual block. It is off by default (``dropout=0.0``), so
    the module is bit-identical to the version without it unless asked otherwise.

    ``dropout_channels=True`` switches to ``Dropout1d``, which drops whole
    feature maps instead of individual positions. On a 1D genomic signal
    neighbouring positions are highly correlated, so element-wise dropout is
    weak -- the surviving neighbours carry almost the same information.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        normalization: str = "rms_batch_norm",
        rms_decay: float = 0.9,
        standardized: bool = True,
        dropout: float = 0.0,
        dropout_channels: bool = True,
    ):
        super().__init__()
        self.norm = make_norm(normalization, in_channels, rms_decay)
        if dropout > 0:
            self.drop = nn.Dropout1d(dropout) if dropout_channels else nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()
        self.conv = make_conv1d(in_channels, out_channels, kernel_size, standardized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.drop(F.gelu(self.norm(x))))


class DNAEmbedder(nn.Module):
    """First layer: wide convolution over one-hot DNA plus one residual block.

        out = Conv1d(4 -> C, k=stem_kernel_size)(x)
        out = out + ConvBlock(C -> C, k=block_kernel_size)(out)
    """

    def __init__(
        self,
        out_channels: int,
        stem_kernel_size: int = 15,
        block_kernel_size: int = 5,
        normalization: str = "rms_batch_norm",
        rms_decay: float = 0.9,
        standardized: bool = True,
        dropout: float = 0.0,
        dropout_channels: bool = True,
    ):
        super().__init__()
        self.stem = make_conv1d(4, out_channels, stem_kernel_size, standardized)
        self.block = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=block_kernel_size,
            normalization=normalization,
            rms_decay=rms_decay,
            standardized=standardized,
            dropout=dropout,
            dropout_channels=dropout_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        return out + self.block(out)


class DownresBlock(nn.Module):
    """One encoder stage: grow channels, two residual conv blocks, then pool.

    The first convolution computes all ``out_channels`` output channels from all
    ``in_channels`` inputs, so its weight is ``[out_channels, in_channels, k]``.
    The residual branch has only ``in_channels``, so it is zero-padded along the
    channel axis before the addition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        pool_stride: int = 2,
        normalization: str = "rms_batch_norm",
        rms_decay: float = 0.9,
        standardized: bool = True,
        dropout: float = 0.0,
        dropout_channels: bool = True,
    ):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError(
                f"DownresBlock cannot shrink channels ({in_channels} -> {out_channels}); "
                "the residual branch is zero-padded, not projected."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channel_pad = self.out_channels - self.in_channels

        self.block1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            normalization=normalization,
            rms_decay=rms_decay,
            standardized=standardized,
            dropout=dropout,
            dropout_channels=dropout_channels,
        )
        self.block2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            normalization=normalization,
            rms_decay=rms_decay,
            standardized=standardized,
            dropout=dropout,
            dropout_channels=dropout_channels,
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.channel_pad:
            # Pad the channel axis with zeros: [x1..xC] -> [x1..xC, 0..0].
            residual = F.pad(residual, (0, 0, 0, self.channel_pad))
        out = self.block1(x) + residual
        out = out + self.block2(out)
        return self.pool(out)


class AlphaGenomeStyleCNNEncoder(nn.Module):
    """Convolutional DNA encoder: (B, S) nucleotide codes -> (B, S / stride, C).

    Deviations from the published AlphaGenome encoder
    -------------------------------------------------
    1. Default ``initial_channels=256`` instead of 768 (and therefore
       256..1024 instead of 768..1536). The exact schedule is reachable by
       passing ``initial_channels=768``.
    2. Intended for windows of ~10^5 bp rather than 1,048,576 bp.
    3. No U-Net decoder and no skip outputs -- the downstream task does not need
       dense per-nucleotide predictions.
    4. Optional gradient checkpointing per stage (on by default), which
       AlphaGenome does not describe.
    5. ``RMSBatchNorm1d`` statistics are averaged across DDP ranks.

    Everything else follows the paper: a k=15 stem over one-hot DNA, residual
    k=5 standardized conv blocks, +128 channels per stage with a zero-padded
    channel residual, and MaxPool(2, 2) after every stage.
    """

    def __init__(
        self,
        initial_channels: int = 256,
        channel_increment: int = 128,
        num_pooling_stages: int = 7,
        num_channel_increments: Optional[int] = None,
        stem_kernel_size: int = 15,
        block_kernel_size: int = 5,
        pool_size: int = 2,
        pool_stride: int = 2,
        normalization: str = "rms_batch_norm",
        rms_batch_norm_decay: float = 0.9,
        use_standardized_convolution: bool = True,
        gradient_checkpointing: bool = True,
        dropout: float = 0.0,
        dropout_channels: bool = True,
    ):
        super().__init__()
        if pool_size != pool_stride:
            raise ValueError(
                f"Non-overlapping pooling is assumed (pool_size == pool_stride), "
                f"got {pool_size} and {pool_stride}"
            )

        self.channels = build_channel_schedule(
            initial_channels=initial_channels,
            channel_increment=channel_increment,
            num_pooling_stages=num_pooling_stages,
            num_channel_increments=num_channel_increments,
        )
        self.num_pooling_stages = int(num_pooling_stages)
        self.pool_stride = int(pool_stride)
        self.total_stride = int(pool_stride) ** int(num_pooling_stages)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.register_buffer("onehot_lut", build_onehot_lookup(), persistent=False)

        self.embedder = DNAEmbedder(
            out_channels=self.channels[0],
            stem_kernel_size=stem_kernel_size,
            block_kernel_size=block_kernel_size,
            normalization=normalization,
            rms_decay=rms_batch_norm_decay,
            standardized=use_standardized_convolution,
            dropout=dropout,
            dropout_channels=dropout_channels,
        )
        self.dropout_p = float(dropout)
        self.blocks = nn.ModuleList(
            DownresBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=block_kernel_size,
                pool_size=pool_size,
                pool_stride=pool_stride,
                normalization=normalization,
                rms_decay=rms_batch_norm_decay,
                standardized=use_standardized_convolution,
                dropout=dropout,
                dropout_channels=dropout_channels,
            )
            for i in range(self.num_pooling_stages)
        )

    @property
    def output_channels(self) -> int:
        return self.channels[-1]

    def output_length(self, sequence_length: int) -> int:
        self.validate_length(sequence_length)
        return sequence_length // self.total_stride

    def validate_length(self, sequence_length: int) -> None:
        if sequence_length % self.total_stride != 0:
            raise ValueError(
                f"Sequence length {sequence_length} is not divisible by "
                f"{self.total_stride} (= {self.pool_stride} ** {self.num_pooling_stages}). "
                f"Pad or trim the window to a multiple of {self.total_stride}, or lower "
                f"num_pooling_stages."
            )

    def one_hot(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, S) integer codes -> (B, 4, S) float one-hot. N/pad -> all zeros."""
        if codes.dim() != 2:
            raise ValueError(f"Expected dna codes of shape (B, S), got {tuple(codes.shape)}")
        lut = self.onehot_lut
        onehot = F.embedding(codes.long(), lut)  # (B, S, 4)
        return onehot.transpose(1, 2)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, S) nucleotide codes -> (B, S // total_stride, output_channels)."""
        batch_size, sequence_length = codes.shape
        self.validate_length(sequence_length)

        x = self.one_hot(codes)
        assert x.shape == (batch_size, 4, sequence_length), x.shape

        x = self.embedder(x)
        assert x.shape == (batch_size, self.channels[0], sequence_length), x.shape

        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            expected = (
                batch_size,
                self.channels[i + 1],
                sequence_length // (self.pool_stride ** (i + 1)),
            )
            assert x.shape == expected, f"stage {i}: {tuple(x.shape)} != {expected}"

        return x.transpose(1, 2).contiguous()

    def parameter_report(self) -> str:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        lines = [f"embedder (1 bp, {self.channels[0]} ch): {count(self.embedder):,}"]
        for i, block in enumerate(self.blocks):
            resolution = self.pool_stride ** (i + 1)
            lines.append(
                f"block {i} ({self.channels[i]} -> {self.channels[i + 1]} ch, "
                f"out {resolution} bp): {count(block):,}"
            )
        lines.append(f"total: {count(self):,}")
        return "\n".join(lines)

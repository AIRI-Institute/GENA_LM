"""Flashzoi / Borzoi as a DNA encoder for the expression models.

Drop-in replacement for :class:`AlphaGenomeStyleCNNEncoder`: takes the same
``(B, S)`` uint8 nucleotide codes the CNN dataset already produces and returns
``(B, out_bins, 1536)``, so ``expression_model_*`` only has to change the input
width of its projection.

Why this is a wrapper rather than a bare ``Borzoi.from_pretrained``
-------------------------------------------------------------------
Six things have to happen that ``borzoi_pytorch`` does not do:

1. **One-hot.** ``conv_dna`` is ``nn.Conv1d(in_channels=4, ...)`` and wants
   ``(B, 4, L)`` float; the dataset hands out ``(B, S)`` uint8 codes. The lookup
   table is shared with the CNN path so N -> all-zeros stays one convention.
2. **Dead heads.** We stop at ``get_embs_after_crop``, so ``final_joined_convs``,
   ``human_head`` and ``mouse_head`` never get a gradient and DDP with
   ``find_unused_parameters=False`` (what the runner uses) hangs. They are
   deleted after loading. With ``tap="transformer"`` the whole U-net upsampling
   path is dead too and goes the same way.
3. **Gradient checkpointing.** There is none upstream. At 32768 bp the tensor
   after ``conv_dna``'s convolution is 512 x 32768 per sample; at the native
   524288 bp it is 512 x 524288 = 537 MB in bf16 for one sample alone.
4. **Frozen BatchNorm.** The conv tower is ``nn.BatchNorm1d`` with pretrained
   running statistics, and the trainer calls ``model.train()`` *on every step*
   (``TrainerAccelerate.step``), which would undo any ``.eval()`` done in
   ``__init__``. Hence the ``train()`` override below. Note that
   ``torch.no_grad()`` does **not** stop BatchNorm from updating its buffers, so
   freezing the tower without freezing BN would still corrupt it.
5. **Frozen tower under ``no_grad``.** The cheapest way to fit a long window is
   to not store the conv tower's activations at all. That cannot be expressed by
   calling ``get_embs_after_crop``: the tower feeds the trainable
   ``horizontal_conv*`` through two skip connections, so the ``detach`` points
   sit in the middle of the function and it has to be re-written.
6. **Crop geometry.** ``out_bins`` has to agree with the dataset's bin coords and
   with the decoder's sequence length, and ``TargetLengthCrop`` silently assumes
   the trim is symmetric. Both are asserted here, once.

Resolutions
-----------
Borzoi pools by 128 in total before the transformer and upsamples twice, so
there are exactly two places worth tapping:

* ``tap="unet"`` (default) -- the U-net output, **32 bp** per bin, ``L / 32``
  bins. This is what Borzoi itself predicts from.
* ``tap="transformer"`` -- the transformer output, **128 bp** per bin,
  ``L / 128`` bins. Skips ``upsampling_unet*`` / ``separable*`` /
  ``horizontal_conv*`` entirely: ~14M fewer parameters and no length-``L/32``
  activations, at 4x coarser resolution.

``pool`` average-pools the chosen output by a further factor, so one decoder
position covers ``bin_size * pool`` bp.
"""

from contextlib import nullcontext
from math import gcd
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Same table the CNN path uses: ACGT -> identity rows, N/IUPAC -> all zeros,
# which is also Borzoi's convention for unknown bases.
from downstream_tasks.expression_prediction.alphagenome_cnn import build_onehot_lookup

# Fixed by the architecture, not tunable: conv_dna pools by 2, res_tower by 16,
# unet1 by 2 (=> 32 bp per U-net bin), and one more pool feeds the transformer
# (=> 128 bp per transformer position).
UNET_BIN_SIZE = 32
TRANSFORMER_BIN_SIZE = 128
# Every max-pool floors, so a length that is not a multiple of 128 makes the
# U-net skip connections disagree by one position and `x + x_unet1` raises.
LENGTH_MULTIPLE = 128


def _is_main_process() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


class FlashzoiEncoder(nn.Module):
    """Pretrained Borzoi/Flashzoi trunk, exposed as ``(B, S) -> (B, out_bins, dim)``.

    Parameters
    ----------
    model_name_or_path:
        ``johahi/flashzoi-replicate-[0-3]``, ``johahi/borzoi-replicate-[0-3]``, or
        a local directory. front3 has no internet, so use a local path there.
    out_bins:
        Positions handed to the decoder, *after* cropping and pooling. The
        decoder sees ``out_bins + 2`` (CLS and SEP).
    pool:
        Extra average pooling factor on top of the tap resolution.
    tap:
        ``"unet"`` (32 bp bins) or ``"transformer"`` (128 bp bins).
    freeze_conv_tower:
        Freeze ``conv_dna``/``res_tower``/``unet1`` and run them under
        ``no_grad``. Large memory win; recommended for the first stage.
    freeze_transformer_blocks:
        Freeze the first K of the 8 transformer blocks.
    freeze_batchnorm:
        Keep every ``BatchNorm1d`` in eval mode, re-applied on each ``train()``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        out_bins: int = 1022,
        pool: int = 1,
        tap: str = "unet",
        pretrained: bool = True,
        freeze_conv_tower: bool = True,
        freeze_transformer_blocks: int = 0,
        freeze_batchnorm: bool = True,
        gradient_checkpointing: bool = True,
        attn_dropout: Optional[float] = None,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()

        try:
            from borzoi_pytorch import Borzoi
            from borzoi_pytorch.config_borzoi import BorzoiConfig
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ImportError(
                "borzoi_pytorch is required for FlashzoiEncoder. Install it with "
                "`pip install --no-deps borzoi-pytorch` -- plain `pip install` pulls "
                "transformers>=4.57.6 and would upgrade the one ModernBERT/Qwen run on."
            ) from exc

        if tap not in ("unet", "transformer"):
            raise ValueError(f"tap must be 'unet' or 'transformer', got {tap!r}")
        if out_bins <= 0 or pool <= 0:
            raise ValueError(f"out_bins and pool must be positive, got {out_bins}, {pool}")

        self.tap = tap
        self.pool = int(pool)
        self.out_bins = int(out_bins)
        self.tap_bin_size = UNET_BIN_SIZE if tap == "unet" else TRANSFORMER_BIN_SIZE
        # bp covered by one decoder position; `total_stride` is the name the
        # CNN encoder uses, kept so the two are interchangeable.
        self.bin_size = self.tap_bin_size * self.pool
        self.total_stride = self.bin_size

        overrides = {}
        if attn_dropout is not None:
            overrides["attn_dropout"] = attn_dropout
        if dropout_rate is not None:
            overrides["dropout_rate"] = dropout_rate
        # bins_to_return only configures borzoi's own TargetLengthCrop, which we
        # bypass (we crop ourselves so both taps behave the same), but keeping it
        # consistent means `self.borzoi.crop` never lies about the geometry.
        if tap == "unet":
            overrides["bins_to_return"] = self.out_bins * self.pool

        self.pretrained = bool(pretrained)
        if self.pretrained:
            self.borzoi = Borzoi.from_pretrained(model_name_or_path, **overrides)
        else:
            # Architecture-only control: same topology, no Borzoi weights.
            #
            # This is NOT naive random init. Verified empirically on this version
            # of borzoi_pytorch: PreTrainedModel.post_init() does not re-run
            # _init_weights here, so the constructor's own scheme survives --
            # convolutions get PyTorch's default kaiming_uniform, and each
            # FlashAttention keeps kaiming_normal(relu) on Wqkv with out_proj
            # zeroed, so every residual branch starts as the identity. That last
            # part is what keeps a from-scratch 8-layer transformer stable.
            cfg = BorzoiConfig.from_pretrained(model_name_or_path)
            for key, value in overrides.items():
                setattr(cfg, key, value)
            self.borzoi = Borzoi(cfg)
        self.output_channels = int(self.borzoi.config.dim)

        # (2) heads and, for the transformer tap, the whole upsampling path are
        # never reached -> no gradient -> DDP(find_unused_parameters=False) hangs.
        dead = ["final_joined_convs", "human_head", "mouse_head", "final_softplus"]
        if tap == "transformer":
            dead += [
                "upsampling_unet0", "upsampling_unet1",
                "separable0", "separable1",
                "horizontal_conv0", "horizontal_conv1",
                "upsample",
            ]
        removed = []
        for name in dead:
            if hasattr(self.borzoi, name):
                delattr(self.borzoi, name)
                removed.append(name)

        self.freeze_conv_tower = bool(freeze_conv_tower)
        self.freeze_batchnorm = bool(freeze_batchnorm)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if not self.pretrained and (self.freeze_conv_tower or self.freeze_batchnorm):
            raise ValueError(
                "pretrained=False has nothing worth freezing: a frozen conv tower "
                "would be frozen noise, and frozen BatchNorm would keep its init "
                "statistics (mean 0, var 1) for the whole run. Set "
                "freeze_conv_tower=False and freeze_batchnorm=False for the "
                "from-scratch control."
            )

        if self.freeze_conv_tower:
            for name in ("conv_dna", "res_tower", "unet1"):
                _set_requires_grad(getattr(self.borzoi, name), False)
            if not self.freeze_batchnorm:
                raise ValueError(
                    "freeze_conv_tower=True with freeze_batchnorm=False corrupts the "
                    "pretrained BatchNorm statistics: torch.no_grad() does not stop "
                    "running_mean/var from being updated."
                )

        n_blocks = len(self.borzoi.transformer)
        k = max(0, min(int(freeze_transformer_blocks), n_blocks))
        for block in list(self.borzoi.transformer)[:k]:
            _set_requires_grad(block, False)
        self.frozen_transformer_blocks = k

        self.register_buffer("onehot_lut", build_onehot_lookup(), persistent=False)

        # (4) applies now and after every model.train() -- see train() below.
        self.train(self.training)

        if _is_main_process():
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            weights = "PRETRAINED" if self.pretrained else "FROM SCRATCH (architecture only)"
            print(
                f"[flashzoi] {model_name_or_path} [{weights}] tap={tap} "
                f"bins={self.out_bins} x {self.bin_size} bp "
                f"= {self.out_bins * self.bin_size:,} bp to the decoder"
            )
            print(f"[flashzoi] removed unused: {removed}")
            print(
                f"[flashzoi] frozen: conv_tower={self.freeze_conv_tower}, "
                f"transformer_blocks={k}/{n_blocks}, batchnorm={self.freeze_batchnorm}"
            )
            print(f"[flashzoi] gradient_checkpointing={self.gradient_checkpointing}")
            print(f"[flashzoi] trainable {trainable:,} / {total:,}")

    # ------------------------------------------------------------------ setup --

    def train(self, mode: bool = True):
        """``nn.Module.train`` plus: BatchNorm stays in eval if frozen.

        The trainer calls ``model.train()`` before every step, so this has to be
        re-applied there rather than once in ``__init__``.
        """
        super().train(mode)
        if self.freeze_batchnorm:
            for m in self.borzoi.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        return self

    # ------------------------------------------------------------- geometry ---

    @property
    def required_length_multiple(self) -> int:
        return LENGTH_MULTIPLE

    @property
    def _length_step(self) -> int:
        """Windows must be a multiple of this: lcm(128, bin_size).

        128 comes from Borzoi's pooling; bin_size from the requirement that the
        window divides evenly into decoder positions, which is what
        ``ExpressionDatasetCNN`` assumes when it lays out the label bins.
        """
        return LENGTH_MULTIPLE * self.bin_size // gcd(LENGTH_MULTIPLE, self.bin_size)

    @property
    def min_input_length(self) -> int:
        """Shortest valid window that can produce ``out_bins`` at this tap."""
        need = self.out_bins * self.bin_size
        step = self._length_step
        return ((need + step - 1) // step) * step

    def validate_length(self, sequence_length: int) -> None:
        if sequence_length % LENGTH_MULTIPLE != 0:
            raise ValueError(
                f"dna window {sequence_length} must be a multiple of {LENGTH_MULTIPLE}: "
                "Borzoi's max-pools floor, and the U-net skip connections would end up "
                "one position apart (`x + x_unet1` raises a shape error)."
            )
        # Everything below is stated on the *pooled* grid -- one entry per decoder
        # position -- because that is the grid ExpressionDatasetCNN lays its label
        # bins on. Checking on the finer tap grid instead lets `pool` even out an
        # odd difference and silently shifts the labels by half a decoder bin.
        if sequence_length % self.bin_size != 0:
            raise ValueError(
                f"dna window {sequence_length} must be divisible by bin_size*pool "
                f"({self.bin_size}); otherwise this crop and the dataset's bin grid "
                f"disagree by up to {self.bin_size // 2} bp. Use a multiple of "
                f"{self._length_step} bp."
            )
        available = sequence_length // self.bin_size
        if available < self.out_bins:
            raise ValueError(
                f"window {sequence_length} bp gives {available} positions at "
                f"{self.bin_size} bp (tap={self.tap}, pool={self.pool}), but out_bins = "
                f"{self.out_bins} were asked for. Need at least {self.min_input_length} bp."
            )
        if (available - self.out_bins) % 2 != 0:
            raise ValueError(
                f"crop must be symmetric: {available} - {self.out_bins} is odd, so the "
                "centre would shift by half a bin relative to the dataset's coordinates."
            )

    def crop_offset_bp(self, sequence_length: int) -> int:
        """bp dropped from each edge. The dataset must apply the same shift.

        Identical by construction to ``ExpressionDatasetCNN.crop_offset``, which
        computes ``((window // stride - out_bins) // 2) * stride`` with
        ``stride == bin_size``.
        """
        available = sequence_length // self.bin_size
        return ((available - self.out_bins) // 2) * self.bin_size

    @staticmethod
    def _center_crop(x: torch.Tensor, target: int) -> torch.Tensor:
        """(B, L, C) -> (B, target, C), symmetric."""
        length = x.shape[1]
        if length == target:
            return x
        trim = (length - target) // 2
        return x[:, trim : length - trim]

    # -------------------------------------------------------------- forward ---

    def _maybe_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Checkpoint a stage if it has anything to get a gradient for.

        Deliberately not gated on ``x.requires_grad``: the one-hot input never
        requires grad, and that guard would switch checkpointing off for
        ``conv_dna``, which is exactly the most expensive stage.
        """
        if not (self.gradient_checkpointing and self.training):
            return module(x)
        if not any(p.requires_grad for p in module.parameters(recurse=True)):
            return module(x)
        return checkpoint(module, x, use_reentrant=False)

    def _warn_once_if_mostly_unknown(self, codes: torch.Tensor) -> None:
        """Catch a window that is mostly all-zero rows, once, on the first batch.

        The usual cause is soft-masked FASTA: hg38 from UCSC is ~50% lowercase,
        and ``encode_nucleotides`` maps anything that is not upper-case ACGT to
        the N code, whose one-hot row is all zeros. The dataset calls ``.upper()``
        before encoding, so this should never fire -- but if it ever does, half
        the input is silently blank and nothing else would tell us.
        """
        if getattr(self, "_unknown_checked", False):
            return
        self._unknown_checked = True
        # Threshold sits between real windows (<1% N outside centromeres/telomeres)
        # and a soft-masked genome read without .upper() (~50% on hg38).
        frac = (codes >= 4).float().mean().item()
        if frac > 0.35 and _is_main_process():
            print(
                f"[flashzoi] WARNING {frac:.0%} of the first window is N/unknown and "
                "encodes to all-zero rows. Soft-masked FASTA read without .upper()?"
            )

    def _conv_tower(self, x: torch.Tensor):
        """(B, 4, L) -> x_unet0 at L/32 (1280 ch), x_unet1 at L/64 (dim ch)."""
        b = self.borzoi
        x = self._maybe_checkpoint(b.conv_dna, x)
        for stage in b.res_tower:
            x = self._maybe_checkpoint(stage, x)
        x_unet0 = x
        for stage in b.unet1:
            x = self._maybe_checkpoint(stage, x)
        return x_unet0, x

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, S) nucleotide codes -> (B, out_bins, dim)."""
        if codes.dim() != 2:
            raise ValueError(f"Expected dna codes of shape (B, S), got {tuple(codes.shape)}")
        sequence_length = codes.shape[1]
        self.validate_length(sequence_length)

        b = self.borzoi
        # Channel order is A,C,G,T with N -> an all-zero row; verified bit for bit
        # against borzoi-pytorch's own wt_seq.npy fixture (see verify_dna_encoding.py).
        x = F.embedding(codes.long(), self.onehot_lut).transpose(1, 2)  # (B, 4, S)
        self._warn_once_if_mostly_unknown(codes)

        # (5) frozen tower: no graph, no stored activations, detach before the
        # trainable skip convolutions.
        with torch.no_grad() if self.freeze_conv_tower else nullcontext():
            x_unet0, x_unet1 = self._conv_tower(x)
        if self.freeze_conv_tower:
            x_unet0, x_unet1 = x_unet0.detach(), x_unet1.detach()

        h = b._max_pool(x_unet1)          # (B, dim, S/128)
        h = h.permute(0, 2, 1)            # (B, S/128, dim)
        for block in b.transformer:
            h = self._maybe_checkpoint(block, h)

        if self.tap == "transformer":
            feats = h                                        # (B, S/128, dim)
        else:
            h = h.permute(0, 2, 1)                           # (B, dim, S/128)
            h = self._maybe_checkpoint(b.upsampling_unet1, h)
            # out-of-place: the upstream `x += x_unet1` is an in-place write on a
            # tensor that checkpoint recomputation may still need.
            h = h + self._maybe_checkpoint(b.horizontal_conv1, x_unet1)
            h = self._maybe_checkpoint(b.separable1, h)
            h = self._maybe_checkpoint(b.upsampling_unet0, h)
            h = h + self._maybe_checkpoint(b.horizontal_conv0, x_unet0)
            h = self._maybe_checkpoint(b.separable0, h)
            feats = h.permute(0, 2, 1)                       # (B, S/32, dim)

        feats = self._center_crop(feats, self.out_bins * self.pool)
        if self.pool > 1:
            feats = F.avg_pool1d(
                feats.transpose(1, 2), kernel_size=self.pool, stride=self.pool
            ).transpose(1, 2)
        return feats                                          # (B, out_bins, dim)

    # --------------------------------------------------------------- report ---

    def parameter_report(self) -> str:
        def count(module):
            return sum(p.numel() for p in module.parameters())

        def trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        b = self.borzoi
        lines = []
        groups = [("conv_dna", b.conv_dna), ("res_tower", b.res_tower), ("unet1", b.unet1)]
        groups.append(("transformer", b.transformer))
        if self.tap == "unet":
            for name in ("horizontal_conv0", "horizontal_conv1",
                         "upsampling_unet0", "upsampling_unet1",
                         "separable0", "separable1"):
                groups.append((name, getattr(b, name)))
        for name, module in groups:
            lines.append(f"{name}: {count(module):,} ({trainable(module):,} trainable)")
        lines.append(f"total: {count(self):,} ({trainable(self):,} trainable)")
        return "\n".join(lines)

"""End-to-end smoke test: real Flashzoi + real decoder + real Qwen on one GPU.

Checks the three things the stub tests cannot:
  * the whole stack runs forward and backward at the real geometry;
  * every parameter with requires_grad actually receives a gradient -- that is
    the precondition for DDP(find_unused_parameters=False), which the runner uses;
  * peak memory per gene, to pick BS.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python .../smoke_flashzoi.py
"""

import os
import sys
import time

import torch

from downstream_tasks.expression_prediction.expression_model_flashzoi import (
    ExpressionCountsFlashzoi,
)

HOME = os.environ.get("GENALM_HOME", "/home/jovyan/shares/SR003.nfs2/aspeedok")
# Local directory where it exists (front3 has no internet), HF id otherwise.
_LOCAL = f"{HOME}/models/flashzoi-replicate-0"
FLASHZOI = os.environ.get("FLASHZOI_MODEL") or (
    _LOCAL if os.path.isdir(_LOCAL) else "johahi/flashzoi-replicate-0"
)
# SMOKE_PRETRAINED=0 measures the from-scratch control, where the conv tower is
# trainable and therefore stores activations.
PRETRAINED = os.environ.get("SMOKE_PRETRAINED", "1") != "0"
WINDOW = 32768
OUT_BINS = 1022
N_KEYS = 14
DESC_LEN = 96


def make_batch(batch_size, device):
    seq_len = OUT_BINS + 2
    labels = torch.zeros(batch_size, N_KEYS, seq_len, 1, device=device)
    labels_mask = torch.zeros(batch_size, N_KEYS, seq_len, 1, dtype=torch.bool, device=device)
    # tpm-only: only position 0 carries a target, as Expression_dataset_v1 does
    labels[:, :, 0, 0] = torch.randn(batch_size, N_KEYS, device=device)
    labels_mask[:, :, 0, 0] = True
    return dict(
        dna_codes=torch.randint(0, 5, (batch_size, WINDOW), device=device),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        labels=labels,
        labels_mask=labels_mask,
        desc_input_ids=torch.randint(1, 1000, (batch_size, N_KEYS, DESC_LEN), device=device),
        desc_attention_mask=torch.ones(batch_size, N_KEYS, DESC_LEN, dtype=torch.long, device=device),
        desc_index=torch.arange(N_KEYS, device=device).repeat(batch_size, 1),
    )


def main():
    if not torch.cuda.is_available():
        print("no CUDA; skipping")
        return 0

    device = "cuda"
    t0 = time.time()
    model = ExpressionCountsFlashzoi(
        hf_model_name_decoder=f"{HOME}/models/decoderdp0.1",
        flashzoi_model_name=FLASHZOI,
        desc_model_name="Qwen/Qwen3-Embedding-0.6B",
        out_bins=OUT_BINS,
        pool=1,
        tap="unet",
        flashzoi_pretrained=PRETRAINED,
        freeze_conv_tower=PRETRAINED,
        freeze_batchnorm=PRETRAINED,
        gradient_checkpointing=True,
        use_tower=False,
        weight=0.05,
    ).to(device)
    print(f"[smoke] built in {time.time() - t0:.1f}s (pretrained={PRETRAINED})")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] parameters: {total:,} total, {trainable:,} trainable")

    # The unmodified runner reads model.cnn for its parameter-count log line.
    assert model.cnn is model.dna_encoder, "cnn alias missing -> runner crashes"
    print(f"[smoke] runner alias model.cnn ok: {sum(p.numel() for p in model.cnn.parameters()):,}")

    sizes = [int(x) for x in os.environ.get("SMOKE_BATCH_SIZES", "1,2,4").split(",")]
    for batch_size in sizes:
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = make_batch(batch_size, device)
        try:
            t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            out.loss.float().backward()
            torch.cuda.synchronize()
            dt = time.time() - t
        except torch.cuda.OutOfMemoryError:
            print(f"[smoke] B={batch_size}: OOM")
            torch.cuda.empty_cache()
            continue

        peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"[smoke] B={batch_size} ({batch_size * N_KEYS} gene-cell rows): "
            f"loss={out.loss.item():.4f} logits={tuple(out.logits.shape)} "
            f"peak={peak:.1f} GiB  fwd+bwd={dt:.2f}s"
        )
        assert out.logits.shape == (batch_size * N_KEYS, OUT_BINS + 2, 1)
        assert out.cls_loss is not None and out.other_loss is None, (
            "tpm-only batch should produce cls_loss only"
        )

    # DDP(find_unused_parameters=False) requires every trainable parameter to get
    # a gradient. This is the check that would otherwise fail as a hang on 8 GPUs.
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        print(f"[smoke] FAIL {len(missing)} trainable parameters got no gradient:")
        for n in missing[:20]:
            print("   -", n)
        return 1
    print("[smoke] OK every trainable parameter received a gradient")

    # Frozen trunk must really be frozen.
    frozen_with_grad = [
        n for n, p in model.dna_encoder.borzoi.named_parameters()
        if not p.requires_grad and p.grad is not None
    ]
    assert not frozen_with_grad, frozen_with_grad
    print("[smoke] OK frozen conv tower produced no gradients")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import time
import argparse
import numpy as np
import torch
import gc
import multiprocessing as mp
from transformers import AutoTokenizer
from transformers import ModernBertModel

AMP_DTYPE = torch.float32


def forward_backbone(model, input_ids, attention_mask):
    if hasattr(model, "bert"):
        return model.bert(input_ids=input_ids, attention_mask=attention_mask)
    return model(input_ids=input_ids, attention_mask=attention_mask)


def _try_batch_worker(q, model_id, seq_len, batch_size, gpu, vocab_size):
    try:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")

        model = ModernBertModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            forward_backbone(model, input_ids, attention_mask)

        torch.cuda.synchronize(device)
        q.put(("ok", True))
    except RuntimeError as e:
        msg = str(e).lower()
        if ("out of memory" in msg) or ("illegal memory access" in msg) or ("device-side assert" in msg):
            q.put(("ok", False))
        else:
            q.put(("error", str(e)))
    except Exception as e:
        q.put(("error", str(e)))


def try_batch_size_subprocess(ctx, model_id, seq_len, batch_size, gpu, vocab_size):
    q = ctx.Queue()
    p = ctx.Process(
        target=_try_batch_worker,
        args=(q, model_id, seq_len, batch_size, gpu, vocab_size),
    )
    p.start()
    p.join()
    status, payload = q.get()
    p = None

    torch.cuda.empty_cache()
    gc.collect()

    if status == "error":
        raise RuntimeError(payload)
    return payload


def auto_find_max_batch(ctx, model_id, seq_len, gpu, vocab_size, max_batch, log=True):
    lo = 0
    hi = 1

    if log:
        print("\n[AutoBatch] Phase 1: exponential growth")

    while hi <= max_batch:
        ok = try_batch_size_subprocess(ctx, model_id, seq_len, hi, gpu, vocab_size)
        if log:
            print(f"[AutoBatch] try {hi}: {'OK' if ok else 'OOM'}")
        if not ok:
            break
        lo = hi
        hi *= 2

    hi = min(hi, max_batch + 1)

    if log:
        print(f"[AutoBatch] Bracket found: lo={lo} (OK), hi={hi} (OOM or limit)")
        print("[AutoBatch] Phase 2: binary search")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok = try_batch_size_subprocess(ctx, model_id, seq_len, mid, gpu, vocab_size)
        if log:
            print(f"[AutoBatch] try {mid}: {'OK' if ok else 'OOM'}  -> ", end="")
        if ok:
            lo = mid
            if log:
                print(f"lo={lo}, hi={hi}")
        else:
            hi = mid
            if log:
                print(f"lo={lo}, hi={hi}")

    found = lo
    if found > 8:
        found -= 4
        if log:
            print(f"[AutoBatch] Safety margin: -4 -> {found}")
    if found < 1:
        found = 1

    if log:
        print(f"[AutoBatch] Final batch size: {found}\n")
    return found


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/jovyan/shares/SR003.nfs2/nt_files/moderngena_base")
    parser.add_argument("--tokenizer", type=str, default="AIRI-Institute/gena-lm-bert-base")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2210)
    parser.add_argument("--auto_batch", action="store_true")
    parser.add_argument("--max_batch", type=int, default=100000)
    parser.add_argument("--log_batch_search", action="store_true")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batches_per_run", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = getattr(tokenizer, "vocab_size", 32000)

    if args.auto_batch:
        args.batch_size = auto_find_max_batch(
            ctx=ctx,
            model_id=args.model,
            seq_len=args.seq_len,
            gpu=args.gpu,
            vocab_size=vocab_size,
            max_batch=args.max_batch,
            log=args.log_batch_search,
        )
        print(f"Auto batch size: {args.batch_size}")

    model = ModernBertModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        attn_implementation="sdpa",torch_dtype=AMP_DTYPE,
    ).to(device)
    model.eval()

    # input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), dtype=torch.long, device=device)
    input_ids = torch.randint(50, 51, (args.batch_size, args.seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((args.batch_size, args.seq_len), dtype=torch.bool, device=device)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        for _ in range(args.warmup):
            forward_backbone(model, input_ids, attention_mask)
    torch.cuda.synchronize(device)

    tokens_per_run = args.batch_size * args.seq_len * args.batches_per_run
    k_tokens_per_sec_runs = []

    for i in range(args.n_runs):
        print(i)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            for _ in range(args.batches_per_run):
                forward_backbone(model, input_ids, attention_mask)

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        dt = t1 - t0
        k_tokens_per_sec_runs.append((tokens_per_run / dt) / 1000.0)

    arr = np.array(k_tokens_per_sec_runs)
    print(f"Model: {args.model}")
    print(f"Seq_len: {args.seq_len}, batch_size: {args.batch_size}")
    print(f"Runs: {args.n_runs}, batches/run: {args.batches_per_run}")
    print(f"Throughput: {arr.mean():.2f} ± {arr.std(ddof=0):.2f} K tokens/s")
    print("Per-run K tokens/s:", ", ".join(f"{x:.2f}" for x in arr))

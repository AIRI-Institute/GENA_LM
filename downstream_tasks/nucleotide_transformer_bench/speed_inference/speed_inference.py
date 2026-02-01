import time
import argparse
import numpy as np
import torch
import gc
import multiprocessing as mp
from src.gena_lm.modeling_bert import BertModel
from transformers import AutoTokenizer, AutoModel
import os
import re
import csv
from datetime import datetime, timezone


def _safe_name(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = s.strip("-")
    return (s[:maxlen] if s else "x")


def forward_backbone(model, input_ids, attention_mask):
    if hasattr(model, "bert"):
        return model.bert(input_ids=input_ids, attention_mask=attention_mask)
    return model(input_ids=input_ids, attention_mask=attention_mask)


def _try_batch_worker(q, model_id, seq_len, batch_size, gpu, vocab_size):
    try:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")

        model = BertModel.from_pretrained(model_id, trust_remote_code=True, add_pooling_layer=False).to(device)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

        with torch.inference_mode():
            forward_backbone(model, input_ids, attention_mask)

        torch.cuda.synchronize(device)
        q.put(("ok", True))
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
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
    parser.add_argument("--max_batch", type=int, default=4096)
    parser.add_argument("--log_batch_search", action="store_true")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batches_per_run", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default=None)

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

    model = BertModel.from_pretrained(args.model, trust_remote_code=True, add_pooling_layer=False).to(device)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((args.batch_size, args.seq_len), dtype=torch.bool, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            forward_backbone(model, input_ids, attention_mask)
    torch.cuda.synchronize(device)

    tokens_per_run = args.batch_size * args.seq_len * args.batches_per_run
    k_tokens_per_sec_runs = []

    with torch.inference_mode():
        for i in range(args.n_runs):
            print(i)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

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

    if args.out_csv:
        model_tag = _safe_name(os.path.basename(args.model.rstrip("/")))
        tok_tag = _safe_name(args.tokenizer.split("/")[-1])
        params_tag = (
            f"m-{model_tag}_t-{tok_tag}"
            f"_seq{args.seq_len}_bs{args.batch_size}_gpu{args.gpu}"
            f"_runs{args.n_runs}_bpr{args.batches_per_run}"
            f"_autob{1 if args.auto_batch else 0}"
        )

        base = args.out_csv
        if base.lower().endswith(".csv"):
            base_noext = base[:-4]
        else:
            base_noext = base

        os.makedirs("results", exist_ok=True)
        out_path = os.path.join("results", f"{os.path.basename(base_noext)}_{params_tag}.csv")
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        file_exists = os.path.isfile(out_path)

        ts = datetime.now(timezone.utc).isoformat()
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))

        with open(out_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)


            if not file_exists:
                w.writerow([
                    "timestamp_utc",
                    "kind",               
                    "run_idx",             
                    "k_tokens_s",          
                    "k_tokens_s_std",       
                    "model",
                    "tokenizer",
                    "seq_len",
                    "batch_size",
                    "gpu",
                    "n_runs",
                    "warmup",
                    "batches_per_run",
                    "auto_batch",
                ])

            # Строки по каждому run
            for i, v in enumerate(arr.tolist()):
                w.writerow([
                    ts,
                    "run",
                    i,
                    float(v),
                    "",
                    args.model,
                    args.tokenizer,
                    args.seq_len,
                    args.batch_size,
                    args.gpu,
                    args.n_runs,
                    args.warmup,
                    args.batches_per_run,
                    1 if args.auto_batch else 0,
                ])

            # Итоговая строка mean/std
            w.writerow([
                ts,
                "summary",
                "",
                mean,
                std,
                args.model,
                args.tokenizer,
                args.seq_len,
                args.batch_size,
                args.gpu,
                args.n_runs,
                args.warmup,
                args.batches_per_run,
                1 if args.auto_batch else 0,
            ])

        print(f"Saved CSV: {out_path}")

#!/usr/bin/env bash
set -euo pipefail

MODEL="zhihan1996/DNABERT-2-117M"
TOKENIZER="zhihan1996/DNABERT-2-117M"

SEQ_LENS=(256 512 1024 2048 4096 9192)

MAX_BATCH=100000

N_RUNS=10
WARMUP=3
BATCHES_PER_RUN=5
GPU=1
# BATCH_SIZE=13500

OUT_CSV_BASE="result"

  #  --batch_size "$BATCH_SIZE" \

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  echo "seq_len=${SEQ_LEN}"
  python3 speed_inference_dnabert2.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --seq_len "$SEQ_LEN" \
    --max_batch "$MAX_BATCH" \
    --log_batch_search \
    --auto_batch \
    --n_runs "$N_RUNS" \
    --warmup "$WARMUP" \
    --batches_per_run "$BATCHES_PER_RUN" \
    --gpu "$GPU" \
    --out_csv "$OUT_CSV_BASE"
done

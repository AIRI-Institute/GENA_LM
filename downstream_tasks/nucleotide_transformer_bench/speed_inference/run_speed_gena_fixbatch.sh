#!/usr/bin/env bash
set -euo pipefail

GENALM_HOME="$(cd "$(dirname "$0")/../../.."; pwd)"
export PYTHONPATH="$GENALM_HOME:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="AIRI-Institute/gena-lm-bert-base-t2t"
TOKENIZER="AIRI-Institute/gena-lm-bert-base-t2t"

SEQ_LENS=(128 256 512)

MAX_BATCH=100000

N_RUNS=10
WARMUP=3
BATCHES_PER_RUN=5
GPU=3
BATCH_SIZE=16400

OUT_CSV_BASE="result"

  #  --batch_size "$BATCH_SIZE" \

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  echo "seq_len=${SEQ_LEN}"
  python3 speed_inference.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --seq_len "$SEQ_LEN" \
    --max_batch "$MAX_BATCH" \
    --batch_size "$BATCH_SIZE" \
    --n_runs "$N_RUNS" \
    --warmup "$WARMUP" \
    --batches_per_run "$BATCHES_PER_RUN" \
    --gpu "$GPU" \
    --out_csv "$OUT_CSV_BASE"
done

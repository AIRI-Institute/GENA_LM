#!/usr/bin/env bash
set -euo pipefail

GENALM_HOME="$(cd "$(dirname "$0")/../../.."; pwd)"
export PYTHONPATH="$GENALM_HOME:${PYTHONPATH:-}"

MODEL="AIRI-Institute/gena-lm-bert-base-t2t"
TOKENIZER="AIRI-Institute/gena-lm-bert-base-t2t"

SEQ_LEN=1024
BATCH_SIZE=20
N_RUNS=3
WARMUP=5
BATCHES_PER_RUN=5
GPU=0

OUT_CSV_BASE="results"

python3 speed_inference.py \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --n_runs "$N_RUNS" \
  --warmup "$WARMUP" \
  --batches_per_run "$BATCHES_PER_RUN" \
  --gpu "$GPU" \
  --out_csv "$OUT_CSV_BASE"

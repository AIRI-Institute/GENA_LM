#!/usr/bin/env bash
set -euo pipefail

# MODEL="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
MODEL="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
TOKENIZER="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

SEQ_LENS=(256 512 1024 2048 4096 9192)
MAX_BATCH=100000

N_RUNS=10
WARMUP=3
BATCHES_PER_RUN=5
GPU=5

OUT_CSV_BASE="result"
MIN_TOKEN=0

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  echo "seq_len=${SEQ_LEN}"

  python3 speed_inference_nt.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --seq_len "$SEQ_LEN" \
    --auto_batch \
    --max_batch "$MAX_BATCH" \
    --log_batch_search \
    --n_runs "$N_RUNS" \
    --warmup "$WARMUP" \
    --batches_per_run "$BATCHES_PER_RUN" \
    --gpu "$GPU" \
    --min_token "$MIN_TOKEN" \
    --out_csv "$OUT_CSV_BASE"
done

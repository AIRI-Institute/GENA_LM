#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash run_full_model_inference.sh SPLIT CELL_SET GPU_ID [BATCH_SIZE]"
  echo "Example: bash run_full_model_inference.sh valid json_812 3 256"
  exit 1
fi

SPLIT="$1"
CELL_SET="$2"
GPU_ID="$3"
BATCH_SIZE="${4:-256}"

TASK_ROOT="${TASK_ROOT:-/home/jovyan/dpanc/benchmarking/GENA_LM}"
GENA_HOME="${GENA_HOME:-/home/jovyan/dpanc/GENA_LM/GENA_LM_expression_branch}"
RUN_ONE="$GENA_HOME/downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/run_gena_lm_model_inference.sh"
LOG_DIR="$TASK_ROOT/outputs/logs"

mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/full_model_${SPLIT}_${CELL_SET}.log"
echo "Starting full_model $SPLIT $CELL_SET on physical GPU $GPU_ID"
echo "Log: $LOG"

bash "$RUN_ONE" full_model "$SPLIT" "$CELL_SET" "$GPU_ID" "$BATCH_SIZE" > "$LOG" 2>&1

echo "Finished full_model $SPLIT $CELL_SET"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash run_all_gena_lm_models.sh CELL_SET GPU_ID [BATCH_SIZE]"
  echo "Example: bash run_all_gena_lm_models.sh json_812 4 256"
  exit 1
fi

CELL_SET="$1"
GPU_ID="$2"
BATCH_SIZE="${3:-256}"

TASK_ROOT="${TASK_ROOT:-/home/jovyan/dpanc/benchmarking/GENA_LM}"
GENA_HOME="${GENA_HOME:-/home/jovyan/dpanc/GENA_LM/GENA_LM_expression_branch}"
RUN_ONE="$GENA_HOME/downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/run_gena_lm_model_inference.sh"
LOG_DIR="$TASK_ROOT/outputs/logs"

MODELS=(
  all_datasets_2
  decoder
  dev_loss
  glioma
  len_2048
  modernbert_large
)

SPLITS=(
  valid
  test
)

mkdir -p "$LOG_DIR"

for MODEL_NAME in "${MODELS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    LOG="$LOG_DIR/${MODEL_NAME}_${SPLIT}_${CELL_SET}.log"
    echo "Starting $MODEL_NAME $SPLIT $CELL_SET on physical GPU $GPU_ID"
    echo "Log: $LOG"
    bash "$RUN_ONE" "$MODEL_NAME" "$SPLIT" "$CELL_SET" "$GPU_ID" "$BATCH_SIZE" > "$LOG" 2>&1
    echo "Finished $MODEL_NAME $SPLIT $CELL_SET"
  done
done

echo "All model predictions finished."

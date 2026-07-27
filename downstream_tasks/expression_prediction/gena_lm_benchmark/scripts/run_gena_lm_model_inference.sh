#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: bash run_gena_lm_model_inference.sh MODEL_NAME SPLIT CELL_SET GPU_ID [BATCH_SIZE]"
  echo "Example: bash run_gena_lm_model_inference.sh glioma valid json_812 4 256"
  echo "MODEL_NAME: full_model all_datasets_2 decoder dev_loss glioma len_2048 modernbert_large"
  echo "SPLIT: valid or test"
  echo "CELL_SET: json_14 or json_812"
  exit 1
fi

MODEL_NAME="$1"
SPLIT="$2"
CELL_SET="$3"
GPU_ID="$4"
BATCH_SIZE="${5:-256}"

if [ "$SPLIT" != "valid" ] && [ "$SPLIT" != "test" ]; then
  echo "SPLIT must be valid or test"
  exit 1
fi

if [ "$CELL_SET" != "json_14" ] && [ "$CELL_SET" != "json_812" ]; then
  echo "CELL_SET must be json_14 or json_812"
  exit 1
fi

source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate api

TASK_ROOT="${TASK_ROOT:-/home/jovyan/dpanc/benchmarking/GENA_LM}"
GENA_HOME="${GENA_HOME:-/home/jovyan/dpanc/GENA_LM/GENA_LM_expression_branch}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dpanc/benchmarking/data}"

SCRIPT="$GENA_HOME/downstream_tasks/expression_prediction/inference_example/run_polina_batch_inference.py"
MODEL_DIR="$TASK_ROOT/models/$MODEL_NAME"
OUT_DIR="$TASK_ROOT/predictions_results/$MODEL_NAME"
OUT_CSV="$OUT_DIR/gena_lm_${SPLIT}_${CELL_SET}_predictions.csv"

DNA_MAX_SEQ_LEN=1024

case "$MODEL_NAME" in
  full_model)
    CHECKPOINT="$MODEL_DIR/pytorch_model.bin"
    ;;
  all_datasets_2)
    CHECKPOINT="$MODEL_DIR/pytorch_model.bin"
    ;;
  decoder)
    CHECKPOINT="$MODEL_DIR/pytorch_model.bin"
    ;;
  dev_loss)
    CHECKPOINT="$MODEL_DIR/devloss_pytorch_model.bin"
    ;;
  glioma)
    CHECKPOINT="$MODEL_DIR/glioma_pytorch_model.bin"
    ;;
  len_2048)
    CHECKPOINT="$MODEL_DIR/20260714_2048_best_pytorch_model.bin"
    DNA_MAX_SEQ_LEN=2048
    ;;
  modernbert_large)
    CHECKPOINT="$MODEL_DIR/pytorch_model.bin"
    ;;
  *)
    echo "Unknown MODEL_NAME: $MODEL_NAME"
    exit 1
    ;;
esac

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

mkdir -p "$OUT_DIR" "$TASK_ROOT/outputs/logs"

echo "model: $MODEL_NAME"
echo "split: $SPLIT"
echo "cell set: $CELL_SET"
echo "physical GPU: $GPU_ID"
echo "batch size: $BATCH_SIZE"
echo "DNA max seq len: $DNA_MAX_SEQ_LEN"
echo "checkpoint: $CHECKPOINT"
echo "output: $OUT_CSV"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u "$SCRIPT" \
  --task-root "$TASK_ROOT" \
  --gena-home "$GENA_HOME" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --json-dir "$TASK_ROOT/json_runs/$CELL_SET" \
  --split "$SPLIT" \
  --device cuda:0 \
  --batch-size "$BATCH_SIZE" \
  --dna-max-seq-len "$DNA_MAX_SEQ_LEN" \
  --output "$OUT_CSV"

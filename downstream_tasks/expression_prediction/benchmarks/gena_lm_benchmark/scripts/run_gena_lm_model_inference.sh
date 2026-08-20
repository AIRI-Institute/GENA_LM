#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: bash run_gena_lm_model_inference.sh MODEL_NAME SPLIT CELL_SET GPU_ID [BATCH_SIZE]"
  echo "Example: bash run_gena_lm_model_inference.sh glioma valid json_812 4 256"
  echo "MODEL_NAME: ATAC all_datasets all_datasets2 all_datasets_2 dev_loss full_model glioma len_2048 mult_loss xlarge"
  echo "SPLIT: valid or test"
  echo "CELL_SET: json_14 or json_815"
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
  echo "CELL_SET must be json_14 or json_815"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GENA_HOME="${GENA_HOME:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
TASK_ROOT="${TASK_ROOT:-$GENA_HOME}"
DATA_ROOT="${DATA_ROOT:-$GENA_HOME/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT="$GENA_HOME/downstream_tasks/expression_prediction/inference_example/run_polina_batch_inference.py"
MODEL_DIR="$TASK_ROOT/models/$MODEL_NAME"
OUT_DIR="$TASK_ROOT/predictions_results/$MODEL_NAME"
OUT_CSV="$OUT_DIR/gena_lm_${SPLIT}_${CELL_SET}_predictions.csv"

DNA_MAX_SEQ_LEN=1024

case "$MODEL_NAME" in
  ATAC)
    CHECKPOINT="$MODEL_DIR/ATAC_125000.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/ATAC.yaml"
    ;;
  all_datasets)
    CHECKPOINT="$MODEL_DIR/all_datasets_best.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/all_datasets.yaml"
    ;;
  all_datasets2)
    CHECKPOINT="$MODEL_DIR/all_datasets2_best.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/all_datasets2.yaml"
    ;;
  all_datasets_2)
    CHECKPOINT="$MODEL_DIR/pytorch_model.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/final_02062026.yaml"
    ;;
  dev_loss)
    CHECKPOINT="$MODEL_DIR/devloss_pytorch_model.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/dev_loss.yaml"
    ;;
  full_model)
    CHECKPOINT="$MODEL_DIR/pytorch_model.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/final.yaml"
    ;;
  glioma)
    CHECKPOINT="$MODEL_DIR/glioma_pytorch_model.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/glioma.yaml"
    ;;
  len_2048)
    CHECKPOINT="$MODEL_DIR/20260714_2048_best_pytorch_model.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/final_2048.yaml"
    DNA_MAX_SEQ_LEN=2048
    ;;
  mult_loss)
    CHECKPOINT="$MODEL_DIR/mult_loss_best.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/mult_loss.yaml"
    ;;
  xlarge)
    CHECKPOINT="$MODEL_DIR/xlarge_best.bin"
    EXPERIMENT_CONFIG="$MODEL_DIR/xlarge.yaml"
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

if [ ! -f "$EXPERIMENT_CONFIG" ]; then
  echo "Experiment config not found: $EXPERIMENT_CONFIG"
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
echo "experiment config: $EXPERIMENT_CONFIG"
echo "output: $OUT_CSV"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u "$SCRIPT" \
  --task-root "$TASK_ROOT" \
  --gena-home "$GENA_HOME" \
  --data-root "$DATA_ROOT" \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --json-dir "$TASK_ROOT/json_runs/$CELL_SET" \
  --split "$SPLIT" \
  --device cuda:0 \
  --batch-size "$BATCH_SIZE" \
  --dna-max-seq-len "$DNA_MAX_SEQ_LEN" \
  --output "$OUT_CSV"

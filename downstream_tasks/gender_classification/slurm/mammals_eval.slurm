#!/bin/bash
# One array task = one checkpoint = one GPU.
# Submitted by ../submit_mammals_eval.sh, which fills in --array and --export.
# Not meant to be sbatch'd by hand (MANIFEST/DATA_DIR must be exported first).

#SBATCH --job-name=mammals_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# Adjust to your cluster if the default partition/QoS isn't the right one:
# #SBATCH --partition=<queue-name>
# #SBATCH --qos=<qos-name>

cd "$SLURM_SUBMIT_DIR"

: "${MANIFEST:?MANIFEST not set (path to checkpoints_manifest.tsv)}"
: "${DATA_DIR:?DATA_DIR not set (path to mammals_data_contig_separated/)}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_PER_SAMPLE="${N_PER_SAMPLE:-60_000}"
INFERENCE_RESULT_DIR="${INFERENCE_RESULT_DIR:-mammals_inference_runs/}"
CONDA_ENV="${CONDA_ENV:-gender-cls}"

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
LINE=$(sed -n "${LINE_NO}p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "error: no manifest line $LINE_NO in $MANIFEST" >&2
    exit 1
fi
IFS=$'\t' read -r MODEL_PATH PRETRAINED_CONFIG_NAME MAX_LENGTH EXPERIMENT_NAME <<< "$LINE"

# CUDA_VISIBLE_DEVICES is already set/constrained by Slurm's gres plugin for
# this job step (typically to the single GPU it allocated); we just pass it
# through. Falls back to 0 if the cluster doesn't set it (e.g. a non-gres
# partition with 1 GPU visible by default).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== array task $SLURM_ARRAY_TASK_ID on $SLURMD_NODENAME, GPU(s) $CUDA_VISIBLE_DEVICES ==="
echo "experiment:  $EXPERIMENT_NAME"
echo "checkpoint:  $MODEL_PATH"
echo "backbone:    $PRETRAINED_CONFIG_NAME (max_length=$MAX_LENGTH)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

bash evaluate_all_mammals.sh \
    --pretrained-config-name "$PRETRAINED_CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --experiment-dir "$EXPERIMENT_NAME" \
    --n-per-sample "$N_PER_SAMPLE"

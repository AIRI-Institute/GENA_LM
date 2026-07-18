#!/bin/bash
#SBATCH --job-name=mammals_eval
#SBATCH --partition=rnd
#SBATCH --account=shared
#SBATCH --qos=shared-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -eo pipefail

date

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

source "$HOME/envs/gender/bin/activate"
set -u

export HF_HOME="${HF_HOME:-$HOME/.hf}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-0}"

MANIFEST="${MANIFEST:-$SCRIPT_DIR/checkpoints_manifest.tsv}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/mammals_data_contig_separated}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_PER_SAMPLE="${N_PER_SAMPLE:-60000}"
INFERENCE_RESULT_DIR="${INFERENCE_RESULT_DIR:-mammals_inference_runs/}"

: "${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array job}"
test -f "$MANIFEST"
test -d "$DATA_DIR"

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
LINE="$(sed -n "${LINE_NO}p" "$MANIFEST")"
if [[ -z "$LINE" ]]; then
    echo "No manifest line $LINE_NO in $MANIFEST" >&2
    exit 1
fi

IFS=$'\t' read -r MODEL_PATH PRETRAINED_CONFIG_NAME MAX_LENGTH EXPERIMENT_NAME <<< "$LINE"

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "checkpoint: $MODEL_PATH"
echo "data: $DATA_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<set by Slurm>}"
which python

bash evaluate_all_mammals.sh \
    --pretrained-config-name "$PRETRAINED_CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0}" \
    --experiment-dir "$EXPERIMENT_NAME" \
    --n-per-sample "$N_PER_SAMPLE"

date
echo "Done!"

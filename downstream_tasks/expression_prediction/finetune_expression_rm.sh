#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

TBS=112
BS=16
NP=7  


config_name="v1_qnorm_large_tmpfs"

# GENALM_HOME=$(realpath ..) CUDA_VISIBLE_DEVICES=0    python -m downstream_tasks.expression_prediction.run_expression_finetuning_v1 \
GENALM_HOME=$(realpath ..) horovodrun --gloo -np $NP python -m downstream_tasks.expression_prediction.run_expression_finetuning_v1 \
    --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
    --batch_size $BS \
    --gradient_accumulation_steps $(($TBS / ($BS * $NP)))

echo "done"
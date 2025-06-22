#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=0

TBS=36
BS=36
NP=1

#CUDA_VISIBLE_DEVICES=0 python3 -m downstream_tasks.expression_meta_prediction.run_expression_meta_finetuning_rmt \
GENALM_HOME=$(realpath ..) horovodrun --gloo -np $NP python -m downstream_tasks.expression_prediction.run_expression_finetuning_v1 \
    --experiment_config "downstream_tasks/expression_prediction/configs/run_config_Expression_dataset_v1.yaml" \
    --batch_size $BS \
    --gradient_accumulation_steps $(($TBS / ($BS * $NP)))

echo "done"
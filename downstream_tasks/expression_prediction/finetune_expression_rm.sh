#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=0,1

TBS=4
BS=2  
NP=2    

config_name="run_config_Expression_dataset_v1"
#config_name="run_config_Expression_dataset_full_but_small_valid"

#CUDA_VISIBLE_DEVICES=0 python3 -m downstream_tasks.expression_meta_prediction.run_expression_meta_finetuning_rmt \
horovodrun --gloo -np $NP python -m downstream_tasks.expression_prediction.run_expression_finetuning_v1 \
    --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
    --batch_size $BS \
    --gradient_accumulation_steps $(($TBS / ($BS * $NP)))

echo "done"
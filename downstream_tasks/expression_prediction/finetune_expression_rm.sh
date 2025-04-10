#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=0,1

TBS=32
BS=16  
NP=2    

#CUDA_VISIBLE_DEVICES=0 python3 -m downstream_tasks.expression_meta_prediction.run_expression_meta_finetuning_rmt \
horovodrun --gloo -np $NP python -m downstream_tasks.expression_prediction.run_expression_finetuning \
    --experiment_config "downstream_tasks/expression_prediction/configs/run_config_weight_loss.yaml" \
    --backbone_trainable \
    --batch_size $BS \
    --gradient_accumulation_steps $(($TBS / ($BS * $NP)))

echo "done"

#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

 export CUDA_VISIBLE_DEVICES=2,3,4,5,6


# export TORCH_DISTRIBUTED_DEBUG=DETAIL

#  --multi_gpu \

TBS=20
BS=4
NP=5
GAS=$(( TBS / (BS * NP) ))  

config_name="v1_qnorm_large_human"

GENALM_HOME=$(realpath ..) accelerate launch \
  --main_process_port 29519 \
  --num_processes "$NP" \
  --multi_gpu \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_acc \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"

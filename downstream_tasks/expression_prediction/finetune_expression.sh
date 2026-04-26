#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

 export CUDA_VISIBLE_DEVICES=0,1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

#  --multi_gpu \

TBS=80
BS=2
NP=2
GAS=$(( TBS / (BS * NP) ))  

config_name="final_test"

GENALM_HOME=$(realpath ..) accelerate launch \
  --main_process_port 29515 \
  --multi_gpu \
  --num_processes "$NP" \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"
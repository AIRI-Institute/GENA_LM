#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

 export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

#  --multi_gpu \

TBS=210
BS=2
NP=7
GAS=$(( TBS / (BS * NP) ))  

config_name="loss_test_deviation"

GENALM_HOME=$(realpath ..) accelerate launch \
  --multi_gpu \
  --main_process_port 29515 \
  --num_processes "$NP" \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"
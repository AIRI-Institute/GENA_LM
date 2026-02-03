#!/usr/bin/env bash

set -e

cd ../..

export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export GENALM_HOME="/scratch/tsoies-Expression/GENA_LM"

export CUDA_VISIBLE_DEVICES=0


# export TORCH_DISTRIBUTED_DEBUG=DETAIL

#  --multi_gpu \

TBS=1000 # 100!
BS=5
NP=1  #1! gpu count
GAS=$(( TBS / (BS * NP) ))

config_name="config_final_plus_random_borzoi"

GENALM_HOME=$(realpath ..)

srun -c 10 --mem 200G --time 30-0 -p gpu_A100 --gres gpu:1 accelerate launch \
  --main_process_port 29516 \
  --num_processes "$NP" \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"

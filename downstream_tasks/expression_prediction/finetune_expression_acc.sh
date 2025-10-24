#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=1,3

# --multi_gpu \

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

TBS=4
BS=2
NP=2
GAS=$(( TBS / (BS * NP) ))  

config_name="config_multi_species_rna_geo_artem_min_13"



GENALM_HOME=$(realpath ..) accelerate launch \
  --main_process_port 29516 \
  --num_processes "$NP" \
  --multi_gpu \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_acc \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"

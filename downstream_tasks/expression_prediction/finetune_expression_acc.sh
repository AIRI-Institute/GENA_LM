#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
=======
 export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
>>>>>>> origin/backup-expression-pre-modernbert

# --multi_gpu \

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

<<<<<<< HEAD
#  --multi_gpu \

TBS=112
BS=8
NP=7
GAS=$(( TBS / (BS * NP) ))  

config_name="v1_qnorm_human_modernbert"

GENALM_HOME=$(realpath ..) accelerate launch \
  --main_process_port 29517 \
=======
TBS=140
BS=20
NP=7
GAS=$(( TBS / (BS * NP) ))  

config_name="v1_qnorm_large_human_noqnorm"



GENALM_HOME=$(realpath ..) accelerate launch \
  --main_process_port 29518 \
>>>>>>> origin/backup-expression-pre-modernbert
  --num_processes "$NP" \
  --multi_gpu \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_backbone \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"

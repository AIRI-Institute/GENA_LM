#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Original: 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1  # Modified for 2 GPUs


# export TORCH_DISTRIBUTED_DEBUG=DETAIL

#  --multi_gpu \

LOG_DIR="runs/expression/human_mouse"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/full_${TIMESTAMP}.log"

TBS=2
BS=1
# NP=8  # Original: 8 processes for 8 GPUs
NP=2  # Modified for 2 GPUs
GAS=$(( TBS / (BS * NP) ))  

config_name="human_mouse"

GENALM_HOME=$(realpath ..) accelerate launch \
  --main_process_port 29516 \
  --num_processes "$NP" \
  --multi_gpu \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS" \
  2>&1 | tee "$LOG_FILE"
echo "done"

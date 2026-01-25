#!/usr/bin/env bash
set -euo pipefail

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES=0

config_name="H3K4me3"
NP=1

CONFIG="downstream_tasks/nucleotide_transformer_bench/configs/${config_name}.yaml"

for FOLD in 0 1 2 3 4 5 6 7 8 9
do
  PORT=$((29500 + FOLD))

  GENALM_HOME="$(realpath ..)" accelerate launch \
    --main_process_port "$PORT" \
    --num_processes "$NP" \
    --module downstream_tasks.nucleotide_transformer_bench.run_finetune \
    --experiment_config "$CONFIG" \
    --valid_fold "$FOLD" 
done

echo "done"

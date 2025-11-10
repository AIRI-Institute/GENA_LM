#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

<<<<<<< HEAD
 export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
=======
export CUDA_VISIBLE_DEVICES=1,3
>>>>>>> 2dfd0cd3d4f07a5a5bc1d2633181cea1c47aef42

# --multi_gpu \

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

<<<<<<< HEAD
TBS=70
BS=10
NP=7
GAS=$(( TBS / (BS * NP) ))  

config_name="v1_qnorm_large_mouse_ATAC_seq_3_dataset"
=======
TBS=4
BS=2
NP=2
GAS=$(( TBS / (BS * NP) ))  

config_name="config_multi_species_rna_geo_artem_min_13"
>>>>>>> 2dfd0cd3d4f07a5a5bc1d2633181cea1c47aef42



GENALM_HOME=$(realpath ..) accelerate launch \
  --main_process_port 29518 \
  --num_processes "$NP" \
  --multi_gpu \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_acc \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"

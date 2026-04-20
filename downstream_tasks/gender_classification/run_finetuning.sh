#!/bin/bash


# TODO add cuda_visible_devices as the parameter is not used in the script !!

# Define arguments for the script
N_CHUNKS=16
CHUNK_SIZE=3072
FORCE_SAMPLING_FROM_Y=false
FREEZE_BACKBONE=false
# CHRY_NAME=Ys
CHRY_NAME=Y
CHRY_RATIO="${CHRY_RATIO:-0.25}"
DATA_PATH="${DATA_PATH:-/disk/10tb/home/chepurova/chepurova/mammals_data_contig_separated/}"

LR=1e-05
TBS=128
PER_DEVICE_BATCH_SIZE=8
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

EXP_PATH="./runs/mammals_contig_separated_modern_gena_${N_CHUNKS}x${CHUNK_SIZE}_bs_${TBS}_lr_${LR}_${CHRY_NAME}"

if [ -n "$CHRY_RATIO" ]; then
  EXP_PATH="${EXP_PATH}_chrY_ratio_${CHRY_RATIO}"
fi


if [ "$FORCE_SAMPLING_FROM_Y" = true ]; then
  EXP_PATH="${EXP_PATH}_force_y"
fi

if [ "$FREEZE_BACKBONE" = true ]; then
  EXP_PATH="${EXP_PATH}_freeze"
fi

N=1

EXP_PATH="${EXP_PATH}/run_$N"

# ---- Redirect temp & cache dirs ----
TMP_DIR="${TMP_DIR:-/disk/10tb/home/chepurova/bigger_tmp}"
export TMPDIR=$TMP_DIR
export TEMP=$TMP_DIR
export TMP=$TMP_DIR

# conda activate dna-lm 
# Execute the script using accelerate for parallel processing
accelerate launch \
  --main_process_port $((29500+N_CHUNKS*100+CHUNK_SIZE+TBS+N+1)) \
  --num_processes $NP \
  --mixed_precision bf16 \
  --config_file default_config.yaml \
  ./train.py \
  --exp_path $EXP_PATH \
  --n_chunks $N_CHUNKS \
  --chunk_size $CHUNK_SIZE \
  --chrY_name $CHRY_NAME \
  --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --learning_rate $LR \
  --warmup_steps 1000 \
  --max_steps 150000 \
  --eval_steps 250 \
  --early_stopping_patience 5000 \
  --data_path "$DATA_PATH" \
  $( [ "$FORCE_SAMPLING_FROM_Y" = true ] && echo "--force_sampling_from_y" ) \
  $( [ "$FREEZE_BACKBONE" = true ] && echo "--freeze_backbone" ) \
  $( [ -n "$CHRY_RATIO" ] && echo "--chrY_ratio $CHRY_RATIO" )
  # --init_checkpoint ./runs/mammals_contig_separated_16x3072_bs_128_lr_1e-05_Y_chrY_ratio_0.25/run_3/checkpoint-18500/model.safetensors \

  


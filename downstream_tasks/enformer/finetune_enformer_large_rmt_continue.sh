#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=enformer

# BASE_MODEL=bert_base_512_lastln_t2t_1000G_bs256_lr_1e-04_linear_fp16
# BASE_MODEL=bert_base_512_t2t_1000G_multi_from_1M_bs256_lr_1e-04_fp16
# BASE_MODEL=bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16
# BASE_CKPTS=(model_500000 model_1000000 model_2000000)
# BASE_CKPTS=(model_2000000)
BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16_enformer
BASE_CKPTS=(run_1)
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
# CONFIG=./data/configs/L12-H768-A12-V32k-preln-lastln.json
# CONFIG=./data/configs/L12-H768-A12-V32k-preln.json
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json

# SCHEDULER=constant_with_warmup
SCHEDULER=cosine
ITERS=200000
PATIENCE=40
LR=1e-04
OPT=AdamW
WD=0.0

TBS=128
BS=1

# 47 segments in task
# TASK
INPUT_SEQ_LEN=12120
BINS_PER_SAMPLE=456
MAX_N_SEGMENTS=24
# RMT
INPUT_SIZE=512  # segment length
MAX_BINS_PER_SEGMENT=19
MEMORY_SIZE=5
BPTT=-1

AUGS=1
MIX_LENGTH=0.5

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/kuratov/t5-experiments/runs

for N in 1
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
for LR in 1e-04 5e-05 2e-05
do
for SCHEDULER in cosine #constant_with_warmup
do
BASE_CKPT=${BASE_CKPTS[i]}
rmt_params=rmt_seglen_${INPUT_SIZE}_len${INPUT_SEQ_LEN}_maxnsegm_${MAX_N_SEGMENTS}_bins${BINS_PER_SAMPLE}_binspersegm_${MAX_BINS_PER_SEGMENT}_msz_${MEMORY_SIZE}_bptt${BPTT}_mix_${MIX_LENGTH}_augs_${AUGS}
MODEL_PATH=./runs/${TASK}/${BASE_MODEL}/${BASE_CKPT}/${rmt_params}_lr${LR}_${OPT}_${SCHEDULER}_wd${WD}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.enformer.run_enformer_finetuning_rmt \
        --data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_train.h5 \
        --valid_data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_valid.h5 \
        --test_data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_test.h5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --backbone_cls downstream_tasks.enformer.enformer_model:BertForEnformer \
        --model_cls src.gena_lm.modeling_rmt:RMTEncoderForEnformer \
        --init_checkpoint ${HOME_PATH}/kuratov/dnalm/runs/enformer/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16_enformer/run_1/rmt_seglen_512_len8080_maxnsegm_16_bins304_binspersegm_19_msz_5_bptt-1_mix_0.5_augs_1_lr1e-04_AdamW_cosine_wd0.0_p40_bs128_it200000/run_1/model_best.pth \
        --input_seq_len $INPUT_SEQ_LEN --bins_per_sample $BINS_PER_SAMPLE --data_n_workers 2 \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --max_bins_per_segment $MAX_BINS_PER_SEGMENT \
        --mixed_length_ratio $MIX_LENGTH \
        --use_augs $AUGS \
        --backbone_trainable \
        --bptt_depth -1 \
        --n_valid_samples 2000 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 5000 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_iteration --reset_optimizer \
        --log_interval 250 --valid_interval 500 --early_stopping_patience $PATIENCE \
        --optimize_metric pearson_corr_enformer_statefull --optimize_mode max --save_best \
        --seed $(($N+100*$MAX_N_SEGMENTS+42))
#find $MODEL_PATH | grep .pth | xargs -l rm -rf
done
done
done
done
echo "done"
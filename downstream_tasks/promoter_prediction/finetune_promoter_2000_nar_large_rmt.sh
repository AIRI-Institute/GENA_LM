#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

BASE_MODEL=rmt_bert_large_lastln_t2t_1000G_seglen_512_len_3992_maxnsegm_8_msz_10_bptt-1_mix_0.5_bs256_lr_2e-05_wd_1e-04_fp32_from_4_segm
BASE_CKPTS=(model_80000)
# TOKENIZER=./data/tokenizers/human/BPE_32k/
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json

OPT=AdamW
SCHEDULER=constant_with_warmup
TASK=check_promoters/NAR_16000_gap

LEN=2000

ITERS=10000
TBS=256
BS=64
PATIENCE=7
WD=0.0
LR=1e-04
CLIP_NORM=10000  # 10000 is like no clipping
BPE_DROPOUT=0.0
BODY_LR_MULT=0.1

# RMT
INPUT_SIZE=512  # segment length
MAX_N_SEGMENTS=1
MEMORY_SIZE=10

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/t5-experiments/runs

for N in 1 2 3 4 5
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
BASE_CKPT=${BASE_CKPTS[i]}
for LR in 2e-05 5e-05
do
for BODY_LR_MULT in 1
do
MODEL_PATH=./runs/${TASK}_${LEN}/${BASE_MODEL}/${BASE_CKPT}/lr${LR}_body_m${BODY_LR_MULT}_${OPT}_${SCHEDULER}_wd${WD}_cgn${CLIP_NORM}_bpe${BPE_DROPOUT}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.promoter_prediction.run_promoter_finetuning_rmt \
        --data_path ${DATA_PATH}/downstream_tasks/${TASK}/promoters_and_non_promoters_${LEN}/split_${N}/train \
        --valid_data_path ${DATA_PATH}/downstream_tasks/${TASK}/promoters_and_non_promoters_${LEN}/split_${N}/valid \
        --test_data_path ${DATA_PATH}/downstream_tasks/${TASK}/promoters_and_non_promoters_${LEN}/split_${N}/test \
        --model_path $MODEL_PATH \
        --init_checkpoint ${PRETRAINED_PATH}/${BASE_MODEL}/${BASE_CKPT}.pth \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --backbone_cls src.gena_lm.modeling_bert:BertForSequenceClassification \
        --model_cls src.gena_lm.modeling_rmt:RMTEncoderForSequenceClassification \
        --input_seq_len 512 --data_n_workers 2 \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --backbone_trainable \
        --bptt_depth -1 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 250 \
        --body_lr_multiplier ${BODY_LR_MULT} \
        --optimizer ${OPT} --weight_decay $WD \
        --bpe_dropout ${BPE_DROPOUT} \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric f1 --optimize_mode max --save_best \
        --log_interval 100 --valid_interval 100 --early_stopping_patience $PATIENCE \
        --clip_grad_norm $CLIP_NORM \
        --fp16 --apex_opt_lvl O2 \
        --seed $(($N+42))
done
done
done
done
echo "done"

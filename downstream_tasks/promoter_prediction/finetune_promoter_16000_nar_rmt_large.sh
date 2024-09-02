#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

# change to ckpt trained on 2000k
BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
BASE_CKPT=model_1750000
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json

OPT=AdamW
SCHEDULER=constant_with_warmup
TASK=check_promoters/NAR_16000_gap

LEN=16000

ITERS=50000
TBS=128  # total batch size
BS=4  # * grad_acc_steps = per gpu batch size

PATIENCE=10
WD=0.0
LR=5e-05
BODY_LR_MULT=1.0
CLIP_NORM=1.0
BPE_DROPOUT=0.0

# RMT
INPUT_SIZE=512  # segment length
MAX_N_SEGMENTS=6
MEMORY_SIZE=10

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/t5-experiments/runs

for N in 1 2 3 4 5
do
for LR in 2e-05
do
# rmt_params=rmt_seglen_${INPUT_SIZE}_msz_${MEMORY_SIZE}_sum_loss
rmt_params=rmt_seglen_${INPUT_SIZE}_msz_${MEMORY_SIZE}
BASE_MODEL_PATH=./runs/${TASK}_2000/${BASE_MODEL}/${BASE_CKPT}/lr5e-05_body_m1_AdamW_constant_with_warmup_wd0.0_cgn10000_bpe0.0_p7_bs256_it10000/run_${N}
MODEL_PATH=./runs/${TASK}_${LEN}/${BASE_MODEL}/${BASE_CKPT}/${rmt_params}_lr${LR}_body_m${BODY_LR_MULT}_${OPT}_${SCHEDULER}_wd${WD}_cgn${CLIP_NORM}_bpe${BPE_DROPOUT}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
echo "from $BASE_MODEL_PATH"
horovodrun --gloo -np $NP python -m downstream_tasks.promoter_prediction.run_promoter_finetuning_rmt \
        --data_path ${DATA_PATH}/downstream_tasks/${TASK}/promoters_and_non_promoters_${LEN}/split_${N}/train \
        --valid_data_path ${DATA_PATH}/downstream_tasks/${TASK}/promoters_and_non_promoters_${LEN}/split_${N}/valid \
        --test_data_path ${DATA_PATH}/downstream_tasks/${TASK}/promoters_and_non_promoters_${LEN}/split_${N}/test \
        --model_path $MODEL_PATH \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --backbone_cls src.gena_lm.modeling_bert:BertForSequenceClassification \
        --model_cls src.gena_lm.modeling_rmt:RMTEncoderForSequenceClassification \
        --backbone_checkpoint ${BASE_MODEL_PATH}/model_best.pth \
        --input_seq_len 4096 --data_n_workers 2 \
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
        --seed $(($N+42))
done
done
echo "done"

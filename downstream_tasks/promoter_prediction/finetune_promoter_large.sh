#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

# BASE_MODEL=bert_base_512_bs256_lr_1e-04_fp16
BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
# BASE_MODEL=bert_base_512_t2t_1000G_multi_from_1M_bs256_lr_1e-04_fp16
BASE_CKPTS=(model_500000 model_1000000 model_1500000)
#TOKENIZER=./data/tokenizers/human/BPE_32k/
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json

OPT=AdamW
SCHEDULER=constant_with_warmup
TASK=epdnew_promoter

LEN=300_fxd

ITERS=10000
TBS=128  # total batch size
BS=128  # * grad_acc_steps = per gpu batch size
PATIENCE=10
WD=0.0
LR=5e-05
BODY_LR_MULT=0.1
CLIP_NORM=1.0
BPE_DROPOUT=0.1

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/t5-experiments/runs

for N in 1 2 3 4 5
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
for LR in 1e-04 5e-05
do
for BODY_LR_MULT in 1.0 0.1
do
BASE_CKPT=${BASE_CKPTS[i]}
MODEL_PATH=./runs/${TASK}_${LEN}/${BASE_MODEL}/${BASE_CKPT}/lr${LR}_body_m${BODY_LR_MULT}_${OPT}_${SCHEDULER}_wd${WD}_cgn${CLIP_NORM}_bpe${BPE_DROPOUT}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.promoter_prediction.run_promoter_finetuning \
        --data_path ${DATA_PATH}/downstream_tasks/${TASK}/len_${LEN}/split_${N}/train \
        --valid_data_path ${DATA_PATH}/downstream_tasks/${TASK}/len_${LEN}/split_${N}/valid \
        --test_data_path ${DATA_PATH}/downstream_tasks/${TASK}/len_${LEN}/split_${N}/test \
        --model_path $MODEL_PATH \
        --init_checkpoint ${PRETRAINED_PATH}/${BASE_MODEL}/${BASE_CKPT}.pth \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --model_cls src.gena_lm.modeling_bert:BertForSequenceClassification \
        --input_seq_len 128 --data_n_workers 2 \
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
done
done
echo "done"
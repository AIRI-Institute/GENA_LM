#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=spliceai

BASE_MODEL=rmt_bert_large_lastln_t2t_1000G_seglen_512_len_998_maxnsegm_2_msz_10_bptt-1_bs256_lr_1e-06_wd_1e-04_fp32
# BASE_MODEL=bert_base_512_t2t_1000G_multi_from_1M_bs256_lr_1e-04_fp16
# BASE_CKPTS=(model_500000 model_1000000 model_2000000)
BASE_CKPTS=(model_600000)
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json
# CONFIG=./data/configs/L12-H768-A12-V32k-preln.json

# SCHEDULER=constant_with_warmup
SCHEDULER=cosine
ITERS=50000
PATIENCE=10
LR=5e-05
OPT=AdamW
WD=1e-04

TBS=64
BS=4

INPUT_SEQ_LEN=3024
# RMT
INPUT_SIZE=512  # segment length
MAX_N_SEGMENTS=6
MEMORY_SIZE=10
BPTT=-1

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/t5-experiments/runs

for N in 1 2 3
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
for INPUT_SEQ_LEN in 2994 1996 998
do
BASE_CKPT=${BASE_CKPTS[i]}
rmt_params=rmt_seglen_${INPUT_SIZE}_len${INPUT_SEQ_LEN}_maxnsegm_${MAX_N_SEGMENTS}_msz_${MEMORY_SIZE}_bptt${BPTT}
MODEL_PATH=./runs/${TASK}/${BASE_MODEL}/${BASE_CKPT}/${rmt_params}_lr${LR}_${OPT}_${SCHEDULER}_wd${WD}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.SpliceAI.run_spliceai_finetuning_rmt \
        --data_path ${DATA_PATH}/downstream_tasks/SpliceAI/train.csv.gz \
        --valid_data_path ${DATA_PATH}/downstream_tasks/SpliceAI/valid.csv.gz \
        --test_data_path ${DATA_PATH}/downstream_tasks/SpliceAI/dataset_test_0.csv.gz \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --backbone_cls src.gena_lm.modeling_bert:BertForTokenClassification \
        --model_cls src.gena_lm.modeling_rmt:RMTEncoderForTokenClassification \
        --init_checkpoint ${PRETRAINED_PATH}/${BASE_MODEL}/${BASE_CKPT}.pth \
        --input_seq_len $INPUT_SEQ_LEN --data_n_workers 2 \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --backbone_trainable \
        --bptt_depth -1 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1500 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric pr_auc_mean --optimize_mode max --save_best \
        --log_interval 250 --valid_interval 1000 --early_stopping_patience $PATIENCE \
        --seed $(($N+42))
done
done
done
echo "done"

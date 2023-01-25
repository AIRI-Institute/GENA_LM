#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=spliceai

BASE_MODEL=bert_base_sparse_rope_4096_bs256_lr_5e-05_wd0.01_fp16_from_425k
# BASE_CKPTS=(model_500000 model_1000000 model_2000000)
BASE_CKPTS=(model_800000_reset_pooler)
# TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
TOKENIZER=./data/tokenizers/human/BPE_32k/
# CONFIG=./data/configs/L12-H768-A12-V32k-preln-lastln.json
# CONFIG=./data/configs/L12-H768-A12-V32k-preln.json
CONFIG=./data/configs/L12-H768-A12-V32k-L4096-preln-sparse-rope.json

# SCHEDULER=constant_with_warmup
SCHEDULER=cosine
ITERS=50000
PATIENCE=10
LR=5e-05
OPT=AdamW
WD=1e-04

TBS=64
BS=16

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/t5-experiments/runs

for N in 1 2 3
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
BASE_CKPT=${BASE_CKPTS[i]}
MODEL_PATH=./runs/${TASK}/${BASE_MODEL}/${BASE_CKPT}/lr${LR}_${OPT}_${SCHEDULER}_wd${WD}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.SpliceAI.run_spliceai_finetuning \
        --data_path ${DATA_PATH}/downstream_tasks/SpliceAI/train.csv.gz \
        --valid_data_path ${DATA_PATH}/downstream_tasks/SpliceAI/valid.csv.gz \
        --test_data_path ${DATA_PATH}/downstream_tasks/SpliceAI/dataset_test_0.csv.gz \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --model_cls src.gena_lm.modeling_bert:BertForTokenClassification \
        --init_checkpoint ${PRETRAINED_PATH}/${BASE_MODEL}/${BASE_CKPT}.pth \
        --input_seq_len 4096 --data_n_workers 4 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1500 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric pr_auc_mean --optimize_mode max --save_best \
        --log_interval 250 --valid_interval 1000 --early_stopping_patience $PATIENCE \
        --fp16 --apex_opt_lvl O2 \
        --seed $(($N+42))
done
done
echo "done"
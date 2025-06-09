#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=annotation_no_rmt_gena_extra_large_4096_bpe

BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
# BASE_MODEL=bert_base_512_t2t_1000G_multi_from_1M_bs256_lr_1e-04_fp16
# BASE_CKPTS=(model_500000 model_1000000 model_2000000)
BASE_CKPTS=(model_1750000)
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln-small.json
# CONFIG=./data/configs/L12-H768-A12-V32k-preln.json

# SCHEDULER=constant_with_warmup
SCHEDULER=constant_with_warmup
ITERS=500000
PATIENCE=10000
LR=5e-05
OPT=AdamW
WD=0.0

TBS=64
BS=2

INPUT_SEQ_LEN=4096

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
accelerate launch --num_processes $NP --config_file ./downstream_tasks/annotation/accelerate.yaml ./downstream_tasks/annotation/run_annotation_finetuning_no_rmt_accelerate.py \
        --data_path /home/jovyan/shares/SR003.nfs2/10_species/all_genomes.hdf5 \
        --valid_data_path /home/jovyan/shares/SR003.nfs2/val_trans250k_full.hdf5 \
        --test_data_path /home/jovyan/shares/SR003.nfs2/val_trans250k_full.hdf5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --model_cls src.gena_lm.modeling_bert:BertForTokenClassification \
        --input_seq_len $INPUT_SEQ_LEN --data_n_workers 8 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps 1 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 15000 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric loss --optimize_mode min --save_best \
        --log_interval 500 --valid_interval 1500 --early_stopping_patience $PATIENCE \
        --seed $(($N+42))
done
done
echo "done"
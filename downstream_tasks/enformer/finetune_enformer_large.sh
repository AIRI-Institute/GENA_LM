#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=enformer

# BASE_MODEL=bert_base_512_lastln_t2t_1000G_bs256_lr_1e-04_linear_fp16
# BASE_MODEL=bert_base_512_t2t_1000G_multi_from_1M_bs256_lr_1e-04_fp16
BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
# BASE_CKPTS=(model_500000 model_1000000 model_2000000)
BASE_CKPTS=(model_1750000)
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
# CONFIG=./data/configs/L12-H768-A12-V32k-preln-lastln.json
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json

# SCHEDULER=constant_with_warmup
SCHEDULER=cosine
ITERS=1000000
PATIENCE=40
LR=1e-04
OPT=AdamW
WD=0.0

TBS=128
BS=16

AUGS=1

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/kuratov/t5-experiments/runs

for N in 3
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
for LR in 1e-04
do
for SCHEDULER in cosine # constant_with_warmup
do
BASE_CKPT=${BASE_CKPTS[i]}
MODEL_PATH=./runs/${TASK}/${BASE_MODEL}/${BASE_CKPT}/lr${LR}_${OPT}_${SCHEDULER}_wd${WD}_p${PATIENCE}_bs${TBS}_it${ITERS}_augs_${AUGS}/run_${N}
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.enformer.run_enformer_finetuning \
        --data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_train.h5 \
        --valid_data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_valid.h5 \
        --test_data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_test.h5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --model_cls downstream_tasks.enformer.enformer_model:BertForEnformer \
        --init_checkpoint ${PRETRAINED_PATH}/${BASE_MODEL}/${BASE_CKPT}.pth \
        --input_seq_len 512 --bins_per_sample 24 --data_n_workers 4 \
        --use_augs $AUGS \
        --n_valid_samples 5000 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 2500 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --log_interval 250 --valid_interval 500 --early_stopping_patience $PATIENCE \
        --optimize_metric pearson_corr_enformer_statefull --optimize_mode max --save_best \
        --seed $(($N+42))
#find $MODEL_PATH | grep .pth | xargs -l rm -rf
done
done
done
done
echo "done"
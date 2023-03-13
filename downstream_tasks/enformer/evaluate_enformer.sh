#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=enformer

# BASE_MODEL=bert_base_512_lastln_t2t_1000G_bs256_lr_1e-04_linear_fp16
# BASE_MODEL=bert_base_512_t2t_1000G_multi_from_1M_bs256_lr_1e-04_fp16
BASE_MODEL=bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16
# BASE_CKPTS=(model_500000 model_1000000 model_2000000)
BASE_CKPTS=(model_2000000)
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
# CONFIG=./data/configs/L12-H768-A12-V32k-preln-lastln.json
CONFIG=./data/configs/L12-H768-A12-V32k-preln.json

BS=128

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
EXP_PATH=${HOME_PATH}/dnalm/runs/enformer
# --data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_train.h5 \

python -m downstream_tasks.enformer.run_enformer_evaluation \
        --valid_data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_valid.h5 \
        --test_data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_test.h5 \
        --experiment_cfg ${EXP_PATH}/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_2000000/lr2e-04_AdamW_cosine_wd0.0_p15_bs64_it100000/run_4/config.json \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --model_cls downstream_tasks.enformer.enformer_model:BertForEnformer \
        --init_checkpoint ${EXP_PATH}/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_2000000/lr2e-04_AdamW_cosine_wd0.0_p15_bs64_it100000/run_4/model_best.pth \
        --input_seq_len 512 --bins_per_sample 24 --data_n_workers 2 \
        --n_samples -1 \
        --batch_size $BS
echo "done"
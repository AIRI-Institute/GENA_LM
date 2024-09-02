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
BASE_CKPT=model_2000000
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
# CONFIG=./data/configs/L12-H768-A12-V32k-preln-lastln.json
# CONFIG=./data/configs/L12-H768-A12-V32k-preln.json
# CONFIG=./data/configs/L12-H768-A12-V32k-preln.json
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json

BS=16

INPUT_SEQ_LENS=(505 1010 2020 4040 8080 12120 16160 24240) # (512 1024 2048 4096 8192 16384 24576)
BINS_PER_SAMPLES=(19 38 76 152 304 456 608 912)
CTX_BINS_PER_SAMPLES=(9 18 36 72 144 216 288 432) #(4 8 16 32 64 96 128 192) #(15 30 60 120 240 360 480 720)
# INPUT_SEQ_LENS=(16160 24240 32320)
# BINS_PER_SAMPLES=(608 912 1216)
# INPUT_SEQ_LENS=(24576)
# BINS_PER_SAMPLES=(896)
# RMT
INPUT_SIZE=512  # segment length
MAX_N_SEGMENTS=64
MAX_BINS_PER_SEGMENT=19

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
EXP_PATH=${HOME_PATH}/kuratov/dnalm/runs/enformer
# RUN_PATH=${EXP_PATH}/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_2000000/rmt_seglen_512_len1024_maxnsegm_4_bins38_binspersegm_19_msz_5_bptt-1_lr5e-05_AdamW_cosine_wd0.0_p15_bs64_it50000/run_2
# RUN_PATH=${EXP_PATH}/${BASE_MODEL}/${BASE_CKPT}/lr3e-04_AdamW_cosine_wd0.0_p15_bs64_it100000_dbg/run_1
#RUN_PATH=${EXP_PATH}/rmt_bert_large_lastln_t2t_1000G_seglen_512_len_998_maxnsegm_2_msz_10_bptt-1_bs256_lr_1e-06_wd_1e-04_fp32/model_600000/rmt_seglen_512_len1024_maxnsegm_10_bins38_binspersegm_19_msz_10_bptt-1_lr1e-04_AdamW_cosine_wd0.0_p10_bs64_it100000/run_1
# RUN_PATH=${EXP_PATH}/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_2000000/rmt_seglen_512_len12120_maxnsegm_24_bins456_binspersegm_19_msz_5_bptt-1_mix_0.5_lr5e-05_AdamW_cosine_wd0.0_p40_bs64_it100000/run_1
RUN_PATH=${EXP_PATH}/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16_enformer/run_1/rmt_seglen_512_len12120_maxnsegm_24_bins456_binspersegm_19_msz_5_bptt-1_mix_0.5_augs_1_lr1e-04_AdamW_cosine_wd0.0_p40_bs128_it200000/run_1/
# --data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_train.h5 \
# --input_seq_len 512 --bins_per_sample 24 --data_n_workers 2 \
# --valid_data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_valid.h5 \
for (( i=0; i<${#INPUT_SEQ_LENS[@]}; i++ ))
do
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[i]}
BINS_PER_SAMPLE=${BINS_PER_SAMPLES[i]}
CTX_BINS_PER_SAMPLE=${CTX_BINS_PER_SAMPLES[i]}
CTX_MODE=left
#--n_context_bins $CTX_BINS_PER_SAMPLE --context_mode $CTX_MODE \
python -m downstream_tasks.enformer.run_enformer_evaluation \
        --test_data_path ${DATA_PATH}/downstream_tasks/enformer/human/h5/human_test.h5 \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --init_checkpoint ${RUN_PATH}/model_best.pth \
        --experiment_cfg ${RUN_PATH}/config.json \
        --input_seq_len $INPUT_SEQ_LEN --bins_per_sample $BINS_PER_SAMPLE \
        --max_n_segments $MAX_N_SEGMENTS \
        --n_context_bins $CTX_BINS_PER_SAMPLE --context_mode $CTX_MODE \
        --n_samples -1 \
        --batch_size $BS
done
echo "done"

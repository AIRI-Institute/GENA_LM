#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=42k_token_pretrain_and_crf_new

BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
BASE_CKPTS=(model_best)


TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L12-H768-A12-V32k-preln-lastln.json
SUB_CONFIG=./data/configs/L12-H768-A12-V32k-preln-lastln-small.json

SCHEDULER=cosine
ITERS=50000
PATIENCE=100
LR=5e-05
OPT=AdamW
WD=1e-04

BS=1

INPUT_SEQ_LEN=42000
LETTER_LEVEL_INPUT_SEQ_LEN=250000
# RMT
INPUT_SIZE=512  # segment length
MAX_N_SEGMENTS=10000
MEMORY_SIZE=5
BPTT=-1

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/t5-experiments/runs

for N in 1 2 3
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
BASE_CKPT=${BASE_CKPTS[i]}
rmt_params=rmt_seglen_${INPUT_SIZE}_len${INPUT_SEQ_LEN}_maxnsegm_${MAX_N_SEGMENTS}_msz_${MEMORY_SIZE}_bptt${BPTT}
MODEL_PATH=./runs/${TASK}/${BASE_MODEL}/${BASE_CKPT}/${rmt_params}_lr${LR}_${OPT}_${SCHEDULER}_wd${WD}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
accelerate launch --num_processes $NP --config_file ./downstream_tasks/annotation/accelerate.yaml ./downstream_tasks/annotation/run_annotation_finetuning_rmt_accelerate_letter_level_crf.py \
        --data_path /home/jovyan/dnalm/downstream_tasks/annotation/data/trans250k_token_atcg_train_crf.hdf5 \
        --valid_data_path /home/jovyan/dnalm/downstream_tasks/annotation/data/trans250k_token_atcg_val_crf.hdf5 \
        --test_data_path /home/jovyan/dnalm/downstream_tasks/annotation/data/trans250k_token_atcg_val_crf.hdf5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG --sub_model_cfg $SUB_CONFIG \
        --backbone_cls src.gena_lm.modeling_bert:BertForLetterLevelTokenClassification \
        --sub_backbone_cls src.gena_lm.modeling_bert:BertForTokenClassification \
        --model_cls src.gena_lm.modeling_rmt:RMTEncoderForLetterLevelTokenClassification \
        --backbone_checkpoint /home/jovyan/dnalm/runs/annotation/bert_base_512_lastln_t2t_1000G_bs256_lr_1e-04_linear_fp16/model_2000000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_cosine_wd1e-04_p10_bs64_it50000/run_1/model_best/pytorch_model.bin \
        --input_seq_len $INPUT_SEQ_LEN --letter_level_input_seq_len $LETTER_LEVEL_INPUT_SEQ_LEN --data_n_workers 2 \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --use_crf \
        --crf_num_classes 7 --load_rmt_model \
        --bptt_depth -1 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps 1 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 50 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric f1_macro_mean --optimize_mode max --save_best \
        --log_interval 100 --valid_interval 200 --early_stopping_patience $PATIENCE \
        --seed $(($N+42))
done
done
echo "done"
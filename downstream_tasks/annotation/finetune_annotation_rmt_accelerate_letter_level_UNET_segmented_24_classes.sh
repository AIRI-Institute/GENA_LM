#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=NeurIPS_gena_base_rmt_15k_BEND_rebuttal

BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
BASE_CKPTS=(model_1750000)


TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L12-H768-A12-V32k-preln-lastln.json

SCHEDULER=constant_with_warmup
ITERS=50000
PATIENCE=10000
LR=5e-05
OPT=AdamW
WD=1e-04

BS=1

INPUT_SEQ_LEN=2000
LETTER_LEVEL_INPUT_SEQ_LEN=15000
# RMT
INPUT_SIZE=512  # segment length
MAX_N_SEGMENTS=10000
MEMORY_SIZE=10
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
accelerate launch --main_process_port=29504 --num_processes $NP --config_file ./downstream_tasks/annotation/accelerate.yaml ./downstream_tasks/annotation/run_annotation_finetuning_rmt_accelerate_letter_level_UNET_segmented_24_classes.py \
        --data_path /home/jovyan/shares/SR003.nfs2/BEND/train_bend.hdf5 \
        --valid_data_path /home/jovyan/shares/SR003.nfs2/BEND/valid_bend.hdf5 \
        --test_data_path /home/jovyan/shares/SR003.nfs2/BEND/test_bend.hdf5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --backbone_cls src.gena_lm.modeling_bert:BertForLetterLevelTokenClassification \
        --model_cls src.gena_lm.modeling_rmt:RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater \
        --backbone_checkpoint /home/jovyan/dnalm/model_hub/rmt_bert_base_lastln_t2t_1000G_seglen_512_len_3992_maxnsegm_8_msz_10_bptt-1_bs256_lr_1e-05_wd_1e-04_fp16_O2/model_best.pth \
        --input_seq_len $INPUT_SEQ_LEN --letter_level_input_seq_len $LETTER_LEVEL_INPUT_SEQ_LEN --data_n_workers 16  \
        --input_size $INPUT_SIZE --unet_sub_model_input_size 8192 \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --bptt_depth -1 \
        --iters $ITERS \
        --backbone_trainable \
        --batch_size $BS --gradient_accumulation_steps 1 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1000 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric mcc_overall --optimize_mode max --save_best \
        --log_interval 500 --valid_interval 1000 --early_stopping_patience $PATIENCE \
        --seed $(($N+42))
done
done
echo "done"

# --backbone_checkpoint /home/jovyan/dnalm/model_hub/rmt_bert_base_lastln_t2t_1000G_seglen_512_len_3992_maxnsegm_8_msz_10_bptt-1_bs256_lr_1e-05_wd_1e-04_fp16_O2/model_best.pth \

# --backbone_trainable \
# --full_checkpoint /home/jovyan/dnalm/runs/annotation_bert_large_shawerma_continued_from_best_5_classes_no_intergenic_combined_4k_bpe_full_up_to_250k_exon_level_choosing_middle_lr_big_wd_UNET_segmented_UNET_repeater_2_step/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len4096_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/pytorch_model.bin

# transformers:BigBirdForTokenClassification
# --submodel_checkpoint /home/jovyan/dnalm/model_hub/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000.pth \
# src.gena_lm.modeling_bert:BertForTokenClassification \
#./data/configs/hf_bigbird_L12-H768-A12-V32k-L4096-small.json #./data/configs/L24-H1024-A16-V32k-preln-lastln.json
# --backbone_checkpoint /home/jovyan/dnalm/runs/annotation_upweighted_edges_large_42k_flash_attention/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-06_AdamW_constant_with_warmup_wd1e-04_p10000_bs64_it500000/run_1/model_best/pytorch_model.bin \
# --submodel_checkpoint /home/jovyan/dnalm/model_hub/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000.pth 

# --full_checkpoint /home/jovyan/dnalm/runs/annotation_2xlarge_letter_level_longer_training_full_length_seam_boost_post_merging_no_rmt_embed_fast/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/pytorch_model.bin \
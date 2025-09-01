#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=annotation_encoder_bert_large_24_classes_GPT2_1024_decoder_4k_with_overlaps_no_loss_weights_with_all_embeddings_in_decoder_avg_hyperparameters_shuffle_starts_featured_continue

BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
BASE_CKPTS=(model_1750000)


TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json
DECODER_CONFIG=./data/configs/GPT2_long_large.json #    ./data/configs/L24-H1024-A16-V32k-preln-lastln-decoder-24.json

SCHEDULER=constant_with_warmup
ITERS=500000
PATIENCE=10000
LR=5e-05
OPT=AdamW
WD=1e-04

BS=1

INPUT_SEQ_LEN=16000
LETTER_LEVEL_INPUT_SEQ_LEN=72000
# RMT
INPUT_SIZE=512  # segment length
MAX_N_SEGMENTS=10000
MEMORY_SIZE=5
BPTT=-1

HOME_PATH=/home/jovyan
DATA_PATH=${HOME_PATH}/data
PRETRAINED_PATH=${HOME_PATH}/t5-experiments/runs

# --num_processes $NP

for N in 1 2 3
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
BASE_CKPT=${BASE_CKPTS[i]}
rmt_params=rmt_seglen_${INPUT_SIZE}_len${INPUT_SEQ_LEN}_maxnsegm_${MAX_N_SEGMENTS}_msz_${MEMORY_SIZE}_bptt${BPTT}
MODEL_PATH=./runs/${TASK}/${BASE_MODEL}/${BASE_CKPT}/${rmt_params}_lr${LR}_${OPT}_${SCHEDULER}_wd${WD}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
accelerate launch --main_process_port=29501 --num_processes $NP --config_file ./downstream_tasks/annotation/accelerate.yaml ./downstream_tasks/annotation/run_annotation_finetuning_rmt_accelerate_decoder.py \
        --data_path /home/jovyan/shares/SR003.nfs2/decoder/gena_decoder_train.hdf5 \
        --valid_data_path /home/jovyan/shares/SR003.nfs2/decoder/gena_decoder_val.hdf5 \
        --test_data_path /home/jovyan/shares/SR003.nfs2/decoder/gena_decoder_val.hdf5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG --decoder_cfg $DECODER_CONFIG \
        --backbone_cls src.gena_lm.modeling_bert:BertForLetterLevelTokenClassification \
        --decoder_cls transformers:GPT2LMHeadModel \
        --model_cls src.gena_lm.modeling_rmt:RMTEncoderDecoderFeatured \
        --full_checkpoint /home/jovyan/dnalm/runs/annotation_encoder_bert_large_24_classes_GPT2_1024_decoder_4k_with_overlaps_no_loss_weights_with_all_embeddings_in_decoder_avg_hyperparameters_shuffle_starts_featured/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len16000_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/pytorch_model.bin \
        --input_seq_len $INPUT_SEQ_LEN --letter_level_input_seq_len $LETTER_LEVEL_INPUT_SEQ_LEN --data_n_workers 16 \
        --input_size $INPUT_SIZE --decoder_input_size 16000 --decoder_look_back_size 8000 \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --bptt_depth -1 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps 1 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1000 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric pr_auc_mean --optimize_mode max --save_best \
        --log_interval 200 --valid_interval 1000 --early_stopping_patience $PATIENCE \
        --seed $(($N+42))
done
done
echo "done"




# --full_checkpoint /home/jovyan/dnalm/runs/annotation_encoder_bert_large_24_classes_GPT2_1024_decoder_4k_with_overlaps_no_loss_weights_with_all_embeddings_in_decoder_avg_hyperparameters/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len4096_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/pytorch_model.bin \

#        --backbone_trainable \

# --encoder_checkpoint /home/jovyan/dnalm/runs/annotation_bert_large_more_intergenic_new_42k_bpe_from_scratch_bug_fix_continue_with_softer_parameters_all_metrics_shawerma/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr1e-05_AdamW_constant_with_warmup_wd1e-05_p10000_bs64_it500000/run_1/model_best/pytorch_model.bin



# transformers:GPT2LMHeadModel
# src.gena_lm.modeling_bert:BertLMHeadModel

# transformers:BigBirdForTokenClassification
# --submodel_checkpoint /home/jovyan/dnalm/model_hub/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000.pth \
# src.gena_lm.modeling_bert:BertForTokenClassification \
#./data/configs/hf_bigbird_L12-H768-A12-V32k-L4096-small.json #./data/configs/L24-H1024-A16-V32k-preln-lastln.json
# --backbone_checkpoint /home/jovyan/dnalm/runs/annotation_upweighted_edges_large_42k_flash_attention/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-06_AdamW_constant_with_warmup_wd1e-04_p10000_bs64_it500000/run_1/model_best/pytorch_model.bin \
# --submodel_checkpoint /home/jovyan/dnalm/model_hub/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000.pth 

# --full_checkpoint /home/jovyan/dnalm/runs/annotation_2xlarge_letter_level_longer_training_full_length_seam_boost_post_merging_no_rmt_embed_fast/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/pytorch_model.bin \
#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=annotation_bert_large_shawerma_no_intergenic_combined_42k_bpe_250k_atcg_UNET_early_concat_atcg_correct_lr_and_wd

BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
BASE_CKPTS=(model_1750000)


TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json

SCHEDULER=constant_with_warmup
ITERS=500000
PATIENCE=10000
LR=5e-05
OPT=AdamW
WD=1e-04

BS=1

INPUT_SEQ_LEN=42000
LETTER_LEVEL_INPUT_SEQ_LEN=250100
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
accelerate launch --num_processes $NP --config_file ./downstream_tasks/annotation/accelerate.yaml ./downstream_tasks/annotation/run_annotation_finetuning_rmt_accelerate_letter_level_UNET.py \
        --data_path /home/jovyan/shares/SR003.nfs2/shawerma/many_genomes_mane_nt_tib.hdf5 \
        --valid_data_path /home/jovyan/shares/SR003.nfs2/mane_no_intergenic_combined/mane_transcript_val_dataset_max_exon_cds.hdf5 \
        --test_data_path /home/jovyan/shares/SR003.nfs2/mane_no_intergenic_combined/mane_transcript_val_dataset_max_exon_cds.hdf5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --backbone_cls src.gena_lm.modeling_bert:BertForLetterLevelTokenClassification \
        --model_cls src.gena_lm.modeling_rmt:RMTEncoderForLetterLevelTokenClassificationUNET \
        --backbone_checkpoint /home/jovyan/dnalm/runs/annotation_bert_large_shawerma_no_intergenic_combined_42k_bpe_full_up_to_250k_exon_cds_choosing/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr1e-05_AdamW_constant_with_warmup_wd1e-05_p10000_bs64_it500000/run_1/model_best/pytorch_model.bin \
        --input_seq_len $INPUT_SEQ_LEN --letter_level_input_seq_len $LETTER_LEVEL_INPUT_SEQ_LEN --data_n_workers 16 \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --backbone_trainable \
        --bptt_depth -1 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps 1 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1000 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric pr_auc_exon_cds --optimize_mode max --save_best \
        --log_interval 100 --valid_interval 1000 --early_stopping_patience $PATIENCE \
        --seed $(($N+42))
done
done
echo "done"

# transformers:BigBirdForTokenClassification
# --submodel_checkpoint /home/jovyan/dnalm/model_hub/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000.pth \
# src.gena_lm.modeling_bert:BertForTokenClassification \
#./data/configs/hf_bigbird_L12-H768-A12-V32k-L4096-small.json #./data/configs/L24-H1024-A16-V32k-preln-lastln.json
# --backbone_checkpoint /home/jovyan/dnalm/runs/annotation_upweighted_edges_large_42k_flash_attention/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-06_AdamW_constant_with_warmup_wd1e-04_p10000_bs64_it500000/run_1/model_best/pytorch_model.bin \
# --submodel_checkpoint /home/jovyan/dnalm/model_hub/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000.pth 

# --full_checkpoint /home/jovyan/dnalm/runs/annotation_2xlarge_letter_level_longer_training_full_length_seam_boost_post_merging_no_rmt_embed_fast/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/pytorch_model.bin \
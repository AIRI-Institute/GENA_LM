#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=test_val


TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/

SCHEDULER=constant_with_warmup
ITERS=500000
PATIENCE=10000
LR=5e-05
OPT=AdamW
WD=0.0

BS=1

LETTER_LEVEL_INPUT_SEQ_LEN=4096

for N in 1 2 3
do
for (( i=0; i<3; i++ ))
do
MODEL_PATH=./runs/${TASK}/lr${LR}_${OPT}_${SCHEDULER}_wd${WD}_p${PATIENCE}_bs${TBS}_it${ITERS}/run_${N}
echo $MODEL_PATH
accelerate launch --main_process_port=29501 --num_processes $NP --config_file ./downstream_tasks/annotation/accelerate.yaml ./downstream_tasks/annotation/run_annotation_finetuning_CADUSEUS_intergenic_crf.py \
        --data_path /home/jovyan/shares/SR003.nfs2/decoding_human_mane_transcripts/gena_decoder_human_val_intergenic_2k_mane_transcripts_only.h5 \
        --valid_data_path /home/jovyan/shares/SR003.nfs2/decoding_human_mane_transcripts/gena_decoder_human_val_intergenic_2k_mane_transcripts_only.h5 \
        --test_data_path /home/jovyan/shares/SR003.nfs2/decoding_human_mane_transcripts/gena_decoder_human_val_intergenic_2k_mane_transcripts_only.h5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER \
        --backbone_cls src.gena_lm.modeling_rmt:CADUSEUS_for_token_classification_CRF_fast \
        --letter_level_input_seq_len $LETTER_LEVEL_INPUT_SEQ_LEN --data_n_workers 16 \
        --iters $ITERS \
        --backbone_trainable \
        --batch_size $BS --gradient_accumulation_steps 1 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1500 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric f1_mean --optimize_mode max --save_best \
        --log_interval 1500 --valid_interval 3000 --early_stopping_patience $PATIENCE \
        --seed $(($N+42))
done
done
echo "done"

# --backbone_trainable \

#  --checkpoint /home/jovyan/dnalm/runs/NATURE_CADUSEUS_ph_10k_backbone_trainable_shuffle_starts_24_classes_fast_CRF_complete_random_start_continue_continue_minja_test/lr5e-05_AdamW_constant_with_warmup_wd0.0_p10000_bs_it500000/run_1/model_best/model.safetensors \


# --checkpoint /home/jovyan/dnalm/runs/NeurIPS_CADUSEUS_ps_32k_unfreeze_backbone_shuffle_starts/lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/model.safetensors \

# shares/SR003.nfs2/shawerma/many_genomes_mane_nt_tib.hdf5

# --checkpoint /home/jovyan/dnalm/runs/CADUSEUS_ph_32k_correct_saving/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd0.0_p10000_bs_it500000/run_1/model_best/accelerate_state/model.safetensors \

# --checkpoint /home/jovyan/dnalm/runs/CADUSEUS_ph_131k/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd0.0_p10000_bs_it500000/run_1/model_best/model.safetensors \

# --full_checkpoint /home/jovyan/dnalm/runs/annotation_bert_large_shawerma_continued_from_best_5_classes_no_intergenic_combined_4k_bpe_full_up_to_250k_exon_level_choosing_middle_lr_big_wd_UNET_segmented_UNET_repeater_2_step/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len4096_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/pytorch_model.bin

# transformers:BigBirdForTokenClassification
# --submodel_checkpoint /home/jovyan/dnalm/model_hub/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000.pth \
# src.gena_lm.modeling_bert:BertForTokenClassification \
#./data/configs/hf_bigbird_L12-H768-A12-V32k-L4096-small.json #./data/configs/L24-H1024-A16-V32k-preln-lastln.json
# --backbone_checkpoint /home/jovyan/dnalm/runs/annotation_upweighted_edges_large_42k_flash_attention/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-06_AdamW_constant_with_warmup_wd1e-04_p10000_bs64_it500000/run_1/model_best/pytorch_model.bin \
# --submodel_checkpoint /home/jovyan/dnalm/model_hub/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000.pth 

# --full_checkpoint /home/jovyan/dnalm/runs/annotation_2xlarge_letter_level_longer_training_full_length_seam_boost_post_merging_no_rmt_embed_fast/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr5e-05_AdamW_constant_with_warmup_wd1e-04_p10000_bs_it500000/run_1/model_best/pytorch_model.bin \
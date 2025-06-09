#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK=annotation_bert_large_more_intergenic_new_4k_bpe_from_scratch_bug_fix_continue_with_softer_parameters_all_metrics_shawerma_amt_with_sep_cls

BASE_MODEL=bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16
# BASE_MODEL=bert_base_512_t2t_1000G_multi_from_1M_bs256_lr_1e-04_fp16
# BASE_CKPTS=(model_500000 model_1000000 model_2000000)
BASE_CKPTS=(model_1750000)
TOKENIZER=./data/tokenizers/t2t_1000h_multi_32k/
CONFIG=./data/configs/L24-H1024-A16-V32k-preln-lastln.json
# CONFIG=./data/configs/L12-H768-A12-V32k-preln.json

# SCHEDULER=constant_with_warmup
SCHEDULER=constant_with_warmup
ITERS=500000
PATIENCE=10000
LR=1e-05
OPT=AdamW
WD=1e-05

TBS=64
BS=8

INPUT_SEQ_LEN=4096
# RMT
INPUT_SIZE=507  # segment length
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
accelerate launch --main_process_port=29500 --num_processes $NP --config_file ./downstream_tasks/annotation/accelerate.yaml ./downstream_tasks/annotation/run_annotation_finetuning_amt_accelerate.py \
        --data_path /home/jovyan/shares/SR003.nfs2/more_intergenic_shaurma_no_lncRNA_junction_train.hdf5 \
        --valid_data_path /home/jovyan/shares/SR003.nfs2/new_intergenic_6_classes/datasets/more_intergenic_human_mane_forw_6_labels_val_no_junction_new.hdf5 \
        --test_data_path /home/jovyan/shares/shares/SR003.nfs2/new_intergenic_6_classes/datasets/more_intergenic_human_mane_forw_6_labels_val_no_junction_new.hdf5 \
        --model_path ${MODEL_PATH} \
        --tokenizer $TOKENIZER --model_cfg $CONFIG \
        --memory_cell_cls modeling_amt.language_modeling:AssociativeMemoryCell \
        --recurrent_wrapper_cls modeling_amt.language_modeling:AssociativeRecurrentWrapper \
        --d_mem 64 \
        --layers_attr bert.encoder.layer \
        --k2 -1 \
        --backbone_cls src.gena_lm.modeling_bert:BertForTokenClassification \
        --rmt2amt_checkpoint /home/jovyan/dnalm/runs/annotation_bert_large_more_intergenic_new_42k_bpe_from_scratch_bug_fix_continue_with_softer_parameters_all_metrics_shawerma/bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16/model_1750000/rmt_seglen_512_len42000_maxnsegm_10000_msz_5_bptt-1_lr1e-05_AdamW_constant_with_warmup_wd1e-05_p10000_bs64_it500000/run_1/model_best/pytorch_model.bin \
        --input_seq_len $INPUT_SEQ_LEN --data_n_workers 16 \
        --segment_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --backbone_trainable \
        --bptt_depth -1 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps 1 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1000 \
        --optimizer ${OPT} --weight_decay $WD \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric max_f1_exon_level --optimize_mode max --save_best \
        --log_interval 500 --valid_interval 2000 --early_stopping_patience $PATIENCE \
        --seed $(($N+42))
done
done
echo "done"
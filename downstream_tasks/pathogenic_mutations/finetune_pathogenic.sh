#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

SCHEDULER=constant_with_warmup
BASE_MODEL=bert_base_512_bs256_lr_1e-04_fp16
BASE_CKPTS=(model_500000)
TOKENIZER=./data/tokenizers/human/BPE_32k
LR=1e-04
OPT=AdamW

ITERS=5000
SEQ_LEN=128
TBS=32
BS=16

TASK_NAME='S1A_germline_variant'

for LR in 1e-04 5e-05 1e-05
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
BASE_CKPT=${BASE_CKPTS[i]}
horovodrun --gloo -np $NP python -m downstream_tasks.pathogenic_mutations.run_pathogenic_finetuning \
        --data_path ~/data/genomes/downstream_tasks/pathogenic_mutations/${TASK_NAME}_dataset.tsv \
        --n_split 5 \
        --model_path ./runs/${TASK_NAME}/${BASE_MODEL}/${BASE_CKPT}/lr${LR}_${OPT}_${SCHEDULER}_bs${TBS}_it${ITERS}_len${SEQ_LEN}/ \
        --tokenizer $TOKENIZER --model_cfg ./data/configs/L12-H768-A12-V32k-preln.json \
        --model_cls src.gena_lm.modeling_bert:BertForSequenceClassification \
        --init_checkpoint ~/t5-experiments/runs/${BASE_MODEL}/${BASE_CKPT}.pth \
        --input_seq_len $SEQ_LEN --data_n_workers 2 \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --optimizer ${OPT} --weight_decay 1e-04 \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric f1 --optimize_mode max --save_best \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 --early_stopping_patience 20
done
done
echo "done"
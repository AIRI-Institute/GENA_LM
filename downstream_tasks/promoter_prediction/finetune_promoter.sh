#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

ITERS=2500
TBS=256
BASE_MODEL=bert_base_512_bs256_lr_1e-04_fp16
BASE_CKPTS=(model_500000)
TOKENIZER=./data/tokenizers/human/BPE_32k/
OPT=AdamW
SCHEDULER=constant_with_warmup
TASK=epdnew_promoter
LR=1e-04
LEN=300

for N in 1 2 3 4 5
do
for (( i=0; i<${#BASE_CKPTS[@]}; i++ ))
do
BASE_CKPT=${BASE_CKPTS[i]}
horovodrun --gloo -np $NP python -m downstream_tasks.promoter_prediction.run_promoter_finetuning \
        --data_path /home/jovyan/data/downstream_tasks/${TASK}/len_${LEN}/split_${N}/train \
        --valid_data_path /home/jovyan/data/downstream_tasks/${TASK}/len_${LEN}/split_${N}/valid \
        --test_data_path /home/jovyan/data/downstream_tasks/${TASK}/len_${LEN}/split_${N}/test \
        --model_path ./runs/${TASK}_${LEN}/${BASE_MODEL}/${BASE_CKPT}/lr${LR}_${OPT}_${SCHEDULER}_bs${TBS}_it${ITERS}/run_${N} \
        --init_checkpoint /home/jovyan/t5-experiments/runs/${BASE_MODEL}/${BASE_CKPT}.pth \
        --tokenizer $TOKENIZER --model_cfg ./data/configs/L12-H768-A12-V32k-preln.json \
        --model_cls src.gena_lm.modeling_bert:BertForSequenceClassification \
        --input_seq_len 128 --data_n_workers 2 \
        --iters $ITERS \
        --batch_size 128 --gradient_accumulation_steps 2 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 250 \
        --optimizer ${OPT} --weight_decay 0.0 \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric f1 --optimize_mode max --save_best \
        --log_interval 100 --valid_interval 100 \
        --seed $(($N+42))
done
done
echo "done"
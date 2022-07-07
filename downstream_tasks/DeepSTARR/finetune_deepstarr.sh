#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

ITERS=50000
BASE_MODEL=bert_base_512_bs256_lr_1e-04_fp16
BASE_CKPT=model_500000
OPT=AdamW
SCHEDULER=constant_with_warmup
TASK=deepstarr
LR=5e-05

for N in 1 2 3
do
horovodrun --gloo -np $NP python -m downstream_tasks.DeepSTARR.run_deepstarr_finetuning \
        --data_path /home/jovyan/data/downstream_tasks/DeepSTARR/Sequences_activity_Train.txt \
        --fasta_path /home/jovyan/data/downstream_tasks/DeepSTARR/Sequences_Train.fa \
        --valid_data_path /home/jovyan/data/downstream_tasks/DeepSTARR/Sequences_activity_Val.txt \
        --valid_fasta_path /home/jovyan/data/downstream_tasks/DeepSTARR/Sequences_Val.fa \
        --test_data_path /home/jovyan/data/downstream_tasks/DeepSTARR/Sequences_activity_Test.txt \
        --test_fasta_path /home/jovyan/data/downstream_tasks/DeepSTARR/Sequences_Test.fa \
        --model_path ./runs/${TASK}/${BASE_MODEL}/${BASE_CKPT}/lr${LR}_${OPT}_${SCHEDULER}_bs64_it${ITERS}/run_${N} \
        --init_checkpoint /home/jovyan/t5-experiments/runs/${BASE_MODEL}/${BASE_CKPT}.pth \
        --tokenizer ./data/tokenizers/human/BPE_32k/ --model_cfg ./data/configs/L12-H768-A12-V32k-preln.json \
        --model_cls src.gena_lm.modeling_bert:BertForSequenceClassification \
        --input_seq_len 128 --data_n_workers 2 \
        --iters $ITERS \
        --batch_size 64 --gradient_accumulation_steps 1 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1500 \
        --optimizer ${OPT} --weight_decay 1e-04 \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric pearsonr_mean --optimize_mode max --save_best \
        --data_n_workers 2 \
        --log_interval 250 --valid_interval 1000 \
        --seed $(($N+42))
done
echo "done"
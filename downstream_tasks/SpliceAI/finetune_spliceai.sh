#!/usr/bin/env bash
set -e
# change dir to the repository root
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

SCHEDULER=constant_with_warmup
ITERS=50000
BASE_CKPT=model_500000
LR=1e-04
OPT=AdamW

for N in 1 2 3
do
horovodrun --gloo -np $NP python -m downstream_tasks.SpliceAI.run_spliceai_finetuning \
        --data_path ~/data/genomes/downstream_tasks/SpliceAI/train.csv.gz \
        --valid_data_path ~/data/genomes/downstream_tasks/SpliceAI/valid.csv.gz \
        --test_data_path ~/data/genomes/downstream_tasks/SpliceAI/dataset_test_0.csv.gz \
        --model_path ./runs/bert_base_512_bs256_lr_1e-04_fp16/spliceai/${BASE_CKPT}/lr${LR}_${OPT}_${SCHEDULER}_bs64_it${ITERS}/run_${N} \
        --tokenizer ./data/tokenizers/human/BPE_32k/ --model_cfg ./data/configs/L12-H768-A12-V32k-preln.json \
        --model_cls src.gena_lm.modeling_bert:BertForTokenClassification \
        --init_checkpoint ~/t5-experiments/runs/bert_base_512_bs256_lr_1e-04_fp16/${BASE_CKPT}.pth \
        --input_seq_len 512 --data_n_workers 2 \
        --iters $ITERS \
        --batch_size 8 --gradient_accumulation_steps 2 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 1500 \
        --optimizer ${OPT} --weight_decay 1e-04 \
        --reset_lr --reset_optimizer --reset_iteration \
        --optimize_metric pr_auc_mean --optimize_mode max --save_best \
        --data_n_workers 2 \
        --log_interval 250 --valid_interval 1000
done
echo "done"
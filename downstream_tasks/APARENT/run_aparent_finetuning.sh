CUDA_VISIBLE_DEVICES=0 horovodrun --gloo -np 1 python -m downstream_tasks.APARENT.run_APARENT_finetuning --tokenizer /mnt/10tb/home/shmelev/dnalm/data/tokenizers/t2t_1000h_multi_32k/ --model_cfg /mnt/10tb/home/shmelev/dnalm/data/configs/L12-H768-A12-V32k-preln.json --model_cls src.gena_lm.modeling_bert:BertForAPARENTSequenceRegression --input_seq_len 256 --train_csv /mnt/10tb/home/shmelev/dnalm/downstream_tasks/APARENT/dataset_itself/APARENT_train.csv --test_csv /mnt/10tb/home/shmelev/dnalm/downstream_tasks/APARENT/dataset_itself/APARENT_test.csv --lr 5e-05 --weight_decay 1e-04 --reset_lr --reset_optimizer --reset_iteration --optimize_metric pearsonr2 --optimize_mode max --save_best --data_n_workers 2 --log_interval 250 --valid_interval 1000 --data_n_workers 2 --iters 500000 --batch_size 32 --gradient_accumulation_steps 1 --model_path /mnt/10tb/home/shmelev/dnalm/downstream_tasks/APARENT/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/ --init_checkpoint /mnt/10tb/home/shmelev/dnalm/downstream_tasks/APARENT/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best_from_s3.pth --optimizer AdamW --fp16 --apex_opt_lvl O2 --lr_scheduler constant_with_warmup --num_warmup_steps 1500
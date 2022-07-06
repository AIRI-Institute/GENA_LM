# SpliceAI
Predict splice donor and acceptor sites based on DNA sequence.

> todo: how to get the data

original data split: train/test = 162706/16505 samples

## train/valid split
We use 10% of training set as validation set:
```bash
python split_train_valid.py --data_path ./dataset_train_all.csv.gz --valid_ratio 0.1 --seed 42
```
As result, train/valid/test = 146436/16270/16505.

## Finetuning on SpliceAI
Set paths to the data, set hyperparameters in example script `finetune_spliceai.sh` and run training on two GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 NP=2 ./finetune_spliceai.sh
```

# SpliceAI
Predict splice donor and acceptor sites based on DNA sequence.

## Dataset
The dataset and dataset processing code from original Illumina SpliceAI was obtained from [basespace project 66029966](https://basespace.illumina.com/projects/66029966/about) (note: authorization required). The dataset was processed using modified versions of create_dataset.py and create_datafile.py scripts to return .csv.gz files with sequence instead of .h5 files with 1-hot encoded data.

original data split (train/test = 162706/16505 samples) was not changed.

## train/valid split
Since original data does not contain validation split, we use 10% of training set as validation set:
```bash
python split_train_valid.py --data_path ./dataset_train_all.csv.gz --valid_ratio 0.1 --seed 42
```
As result, we obtained train/valid/test = 146436/16270/16505.

## Finetuning on SpliceAI
Set paths to the data, set hyperparameters in example script `finetune_spliceai.sh` and run training on two GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 NP=2 ./finetune_spliceai.sh
```

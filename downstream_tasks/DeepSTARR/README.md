# DeepSTARR
Predict drosophila enhancer activity based on DNA sequence

Dataset: https://data.starklab.org/almeida/DeepSTARR/Data/

Code: https://github.com/bernardo-de-almeida/DeepSTARR
```bash
# FASTA files with DNA sequences of genomic regions from train/val/test sets
wget 'https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_Train.fa'
wget 'https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_Val.fa'
wget 'https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_Test.fa'

# Files with developmental and housekeeping activity of genomic regions from train/val/test sets
wget 'https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_activity_Train.txt'
wget 'https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_activity_Val.txt'
wget 'https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_activity_Test.txt'
```


data split: train/valid/test = 402296/40570/41186 samples


## Finetuning on DeepSTARR
Set paths to the data, edit hyperparameters in example script `finetune_deepstarr.sh` and run training:
```bash
CUDA_VISIBLE_DEVICES=0 NP=1 ./finetune_deepstarr.sh
```

# DeepSTARR
Predict drosophila enhancer activity based on DNA sequence

Dataset can be downloaded from [Stark lab](https://data.starklab.org/almeida/DeepSTARR/Data/), repo for original DeepSTARR CNN solution [here](https://github.com/bernardo-de-almeida/DeepSTARR)

To download data run:

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

No preporcessing is needed, we use train/valid/test as in the original dataset:
```
train/valid/test = 402296/40570/41186 samples
```

## Finetuning on DeepSTARR
Set paths to the data, edit hyperparameters in example script `finetune_deepstarr.sh` and run training:
```bash
CUDA_VISIBLE_DEVICES=0 NP=1 ./finetune_deepstarr.sh
```

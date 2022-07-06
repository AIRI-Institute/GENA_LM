# DeepSTARR
Predict drosophila enhancer activity based on DNA sequence

> todo: how to get the data

data split: train/valid/test = 402296/40570/41186 samples


## Finetuning on DeepSTARR
Set paths to the data, edit hyperparameters in example script `finetune_deepstarr.sh` and run training:
```bash
CUDA_VISIBLE_DEVICES=0 NP=1 ./finetune_deepstarr.sh
```

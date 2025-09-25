# GENA_LM Setup

## Environment Setup

```bash
conda env create -n bert24 -f bert24_env.yml
conda activate bert24
python -m pip install -r requirements.txt

# for modernbert support, also install FA (provided below for A100):
# pip install "flash_attn==2.6.3" --no-build-isolation

# see https://github.com/minjaf/ModernBERT/tree/main for H100 support and more details
```

## Data Download and Preparation

```bash
bash annotation_setup.sh
python3 prepare_data.py
```

## Train:
```bash
CUDA_VISIBLE_DEVICES=1 GENALM_HOME=$(realpath ../../)  python train_with_accelerate.py --config configs/test.yaml
# for modergena:
MODERNBERT_HOME="/disk/10tb/home/fishman/DNALM/ModernBERT" GENALM_HOME=$(realpath ../../) python train_with_accelerate.py --config configs/modernGENA.yaml
```

## Evaluate:
```bash
CUDA_VISIBLE_DEVICES=0 GENALM_HOME=$(realpath ../../) python evaluate_on_chromosome.py --config configs/eval_on_21.yaml
# T2T config: eval_T2T_chr20.yaml
```


## Plot eval figures and cumpute metrics
```bash
python preds2metric.py --bigwig_path ../../runs/annotation/basic/checkpoint-22250/eval/T2T-CHM13v2/NC_060944.1 --threshold 0.5 --max_k 50
```
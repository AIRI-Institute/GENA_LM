# GENA_LM Setup

## Environment Setup

```bash
conda env create -f environment.yml
conda activate annotation
```

## Data Download and Preparation

```bash
bash annotation_setup.sh
python3 prepare_data.py
```

## Test run
```bash
CUDA_VISIBLE_DEVICES=1 GENALM_HOME=$(realpath ../../)  python train_with_accelerate.py --config configs/test.yaml
```
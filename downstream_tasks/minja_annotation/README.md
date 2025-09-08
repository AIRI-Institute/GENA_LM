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

## Train:
```bash
CUDA_VISIBLE_DEVICES=1 GENALM_HOME=$(realpath ../../)  python train_with_accelerate.py --config configs/test.yaml
```

## Evaluate:
```bash
CUDA_VISIBLE_DEVICES=0 GENALM_HOME=$(realpath ../../) python evaluate_on_chromosome.py --config configs/eval_on_21.yaml
```
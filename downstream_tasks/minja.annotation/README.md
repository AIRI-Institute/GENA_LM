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

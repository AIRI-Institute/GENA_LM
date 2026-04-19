# modernGENA sequence classification example

This example shows how to fine-tune `AIRI-Institute/moderngena-base` for a binary
sequence classification task:

- **positive class:** ENCODE CTCF liver peaks (`ENCFF046XEZ`)
- **negative class:** random genomic windows that do not overlap any CTCF peaks on the same split

The pipeline includes:
1. human genome download from UCSC (`hg38.fa.gz`)
2. ENCODE peak download
3. train/valid CSV generation with `10,000 bp` windows (`±5 kb` around centers)
4. fine-tuning + evaluation with Hugging Face `Trainer` and Hydra config

## Files

- `download_and_prepare_data.sh` - full data prep pipeline
- `prepare_ctcf_liver_dataset.py` - BED + FASTA to `train.csv` / `valid.csv`
- `dataset.py` - simple `torch.utils.data.Dataset` class
- `train.py` - Hydra-driven training/evaluation script
- `configs/config.yaml` - default run configuration

## Environment setup

Use the dedicated conda environment for this example.

From the `GENA_LM` repo root:

```bash
mamba env create -f examples/modernGENA/environment.yml
conda activate moderngena-example
```

## Step 1: Download and prepare data

From the `GENA_LM` repo root:

```bash
bash examples/modernGENA/download_and_prepare_data.sh
```

Optional smoke test (small dataset):

```bash
MAX_SAMPLES=200 SEQUENCE_LENGTH=10000 VALID_CHROMS=chr21 bash examples/modernGENA/download_and_prepare_data.sh
```

This generates:

- `examples/modernGENA/data/processed/train.csv`
- `examples/modernGENA/data/processed/valid.csv`

## Step 2: Train and evaluate

```bash
python examples/modernGENA/train.py
```

Override config values from CLI (Hydra):

```bash
python examples/modernGENA/train.py training_args.num_train_epochs=1 training_args.per_device_train_batch_size=2
```

## Notes

- Default validation split uses `chr21`; training uses all other chromosomes.
- Default sequence window length is `10,000 bp` (approximately suitable for 1024 BPE tokens for modernGENA).
- Early stopping is enabled by default with patience `30` validation runs (`callbacks.early_stopping.early_stopping_patience`).
- `train.py` sets `trust_remote_code=true` to load model code from HF checkpoint.
- `accelerate` is required by current Hugging Face `Trainer` versions.
- Metrics include `accuracy`, `PR AUC`, and `ROC AUC` (AUC metrics are skipped if eval split has one class only).
- Output model/tokenizer are saved under `runs/modernGENA/ctcf_liver_base/best_model`.

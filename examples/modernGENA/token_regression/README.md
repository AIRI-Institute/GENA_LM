# modernGENA token regression (CTCF bigWig)

This task trains a token-level regression model on ENCODE CTCF signal:

- signal source: `ENCFF184SOF.bigWig`
- intervals source: CTCF peaks from `ENCFF046XEZ.bed`
- labels are computed on-the-fly via `pybigwig`
- per-token target = mean signal over token's base-pair span

## Prepare data

From `GENA_LM` repo root:

```bash
bash examples/modernGENA/token_regression/download_and_prepare_data.sh
```

Optional smoke dataset:

```bash
MAX_SAMPLES=200 INTERVAL_LENGTH=10000 SHIFT_BP=5000 TEST_CHROMS=chr21 bash examples/modernGENA/token_regression/download_and_prepare_data.sh
```

Outputs:

- `examples/modernGENA/data/processed/token_regression/train.csv`
- `examples/modernGENA/data/processed/token_regression/test.csv`

## Train

```bash
python examples/modernGENA/token_regression/train.py
```

Example overrides:

```bash
python examples/modernGENA/token_regression/train.py training_args.num_train_epochs=1 training_args.per_device_train_batch_size=2
```

Random-initialized backbone (same pipeline, no pretrained weights):

```bash
python examples/modernGENA/token_regression/train.py model.initialize_from_pretrained=false
```

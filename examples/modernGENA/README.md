# modernGENA task examples

This folder contains shared setup plus two task-specific examples for
`AIRI-Institute/moderngena-base`:

- `sequence_classification/` - CTCF peak-vs-background classification
- `token_regression/` - per-token regression from ENCODE CTCF bigWig signal

Both tasks reuse cached artifacts under `examples/modernGENA/data`:

- `data/raw` - downloaded FASTA, BED, bigWig
- `data/processed` - generated interval tables and task-specific CSV files

## Environment setup

Use the dedicated conda environment for this example.

From the `GENA_LM` repo root:

```bash
conda env create -f examples/modernGENA/environment.yml
conda activate moderngena-example
```

## Run sequence classification

From the `GENA_LM` repo root:

```bash
bash examples/modernGENA/sequence_classification/download_and_prepare_data.sh
python examples/modernGENA/sequence_classification/train.py
```

## Run token regression

From the `GENA_LM` repo root:

```bash
bash examples/modernGENA/token_regression/download_and_prepare_data.sh
python examples/modernGENA/token_regression/train.py
```

## Notes

- Shared data downloads are reused if files already exist in `examples/modernGENA/data/raw`.
- Both tasks use Hydra configs and instantiate components via config nodes.
- Both tasks save checkpoints and TensorBoard logs under `runs/modernGENA/...`.

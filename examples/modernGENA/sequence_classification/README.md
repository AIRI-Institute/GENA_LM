# modernGENA sequence classification (CTCF)

This task fine-tunes `AIRI-Institute/moderngena-base` on binary CTCF peak
classification:

- positive windows overlap ENCODE liver CTCF peaks (`ENCFF046XEZ`)
- negatives are sampled windows with zero peak overlap

## Prepare data

From `GENA_LM` repo root:

```bash
bash examples/modernGENA/sequence_classification/download_and_prepare_data.sh
```

Optional small smoke dataset:

```bash
MAX_SAMPLES=200 SEQUENCE_LENGTH=10000 VALID_CHROMS=chr21 bash examples/modernGENA/sequence_classification/download_and_prepare_data.sh
```

Outputs:

- `examples/modernGENA/data/processed/sequence_classification/train.csv`
- `examples/modernGENA/data/processed/sequence_classification/valid.csv`

## Train

```bash
python examples/modernGENA/sequence_classification/train.py
```

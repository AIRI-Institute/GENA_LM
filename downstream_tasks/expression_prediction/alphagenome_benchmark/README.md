# AlphaGenome Benchmark Scripts

This folder contains the scripts used to run AlphaGenome predictions for the
human test and validation gene sets, then compare predictions with qnorm ground
truth expression.

The workflow was used for two cell-type sets:

- `json14`: small 14-cell-type test set.
- `json812`: larger ontology-based cell-type set.

## 1. Before Running

Work from the GENA_LM repository:

```bash
cd /home/biophysinf/DashaP/benchmarking/GENA_LM/GENA_LM
```

The AlphaGenome benchmark files are here:

```text
downstream_tasks/expression_prediction/alphagenome_benchmark
```

Activate the environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alphagenome
```

Set the AlphaGenome API key in the shell. The key is intentionally not stored in
the config files:

```bash
export ALPHAGENOME_API_KEY="your_api_key_here"
```

The configs expect these input files:

```text
/home/biophysinf/DashaP/benchmarking/data/human.test.forward.csv
/home/biophysinf/DashaP/benchmarking/data/human.test.reverse.csv
/home/biophysinf/DashaP/benchmarking/data/human.valid.forward.csv
/home/biophysinf/DashaP/benchmarking/data/human.valid.reverse.csv
/home/biophysinf/DashaP/benchmarking/data/test_true_human.csv
/home/biophysinf/DashaP/benchmarking/data/valid_true_human.csv
/home/biophysinf/DashaP/benchmarking/data/hg38.fna
```

## 2. Prediction Scripts

### `Alphagenome_prediction_benchmarking.py`

Runs the older/manual 14-cell AlphaGenome benchmark. The ontology mapping is
hard-coded inside the script as `EXPERIMENT_ONTOLOGY_DICT`.

Main launcher:

```bash
sbatch downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/run_alphagenome_supported.sbatch
```

Main config:

```text
downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/config.yaml
```

This version uses 16 kb AlphaGenome intervals in `make_interval`.

### `Alphagenome_prediction_benchmarking_json.py`

Runs the newer JSON-based benchmark. It reads an ontology-to-qnorm mapping JSON
from `ALPHAGENOME_ONTOLOGY_JSON`.

Main launcher:

```bash
sbatch downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/run_alphagenome_json.sbatch
```

Main config:

```text
downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/config_alphagenome_supported.yaml
```

This version supports checkpoint restart through `checkpoint_path` and
`checkpoint_every` in the config.

Example for the 14-cell mapping:

```bash
export ALPHAGENOME_ONTOLOGY_JSON="$PWD/downstream_tasks/expression_prediction/alphagenome_benchmark/data/alphagenome_track_to_qnorm_id_json14.json"
sbatch downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/run_alphagenome_json.sbatch
```

Example for the larger mapping:

```bash
export ALPHAGENOME_ONTOLOGY_JSON="$PWD/downstream_tasks/expression_prediction/alphagenome_benchmark/data/alphagenome_track_to_qnorm_id_first.json"
sbatch downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/run_alphagenome_json.sbatch
```

## 3. Config Fields To Check

Before each run, check:

- `output_dir`: where final predictions and summary are written.
- `checkpoint_path`: where partial predictions are saved.
- `splits`: usually `test` and `valid`.
- `prediction_functions`: usually only `intervals`.
- `save_pred_path`: output file name template.

For long jobs, keep `checkpoint_every: 25` or similar. If the job stops, rerun
the same command and the script will resume from the checkpoint file.

## 4. Output Files

The prediction scripts write:

```text
alphagenome_supported_predictions_test_intervals.csv
alphagenome_supported_predictions_valid_intervals.csv
Alphagenome_benchmark_summary.csv
checkpoint_test_intervals.csv
checkpoint_valid_intervals.csv
```

The prediction CSV format is:

```text
gene_id,ENCFF035CWS,ENCFF083EOC,...
ENSG...,value,value,...
```

## 5. Ontology Helper Scripts

These scripts were used to prepare the mapping between qnorm/GENA cell IDs and
AlphaGenome RNA-seq tracks.

### `dump_alphagenome_rna_metadata.py`

Downloads AlphaGenome human RNA-seq output metadata and writes it to CSV. It
uses `ALPHAGENOME_API_KEY`.

```bash
python downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/dump_alphagenome_rna_metadata.py \
  --out data/alphagenome_human_rna_seq_metadata.csv
```

### `build_alphagenome_ontology_map.py`

Uses ENCODE metadata to map qnorm IDs/accessions to biosample ontology IDs such
as `UBERON:...`, `CL:...`, or `EFO:...`.

```bash
python downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/build_alphagenome_ontology_map.py \
  --qnorm-map /home/biophysinf/DashaP/benchmarking/data/Expression_dataset_v1_csv_file_mappings_qnorm.csv \
  --out data/qnorm_alphagenome_ontology_map.csv \
  --cache data/encode_metadata_cache.json
```

### `make_experiment_ontology_dict.py`

Combines the ENCODE ontology map with AlphaGenome RNA-seq metadata. It writes a
support summary and a Python dictionary with qnorm ID to ontology mappings.

```bash
python downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/make_experiment_ontology_dict.py \
  --ontology-map data/qnorm_alphagenome_ontology_map.csv \
  --alphagenome-rna-metadata data/alphagenome_human_rna_seq_metadata.csv \
  --dict-py data/experiment_ontology_dict.py \
  --summary-csv data/qnorm_alphagenome_support_summary.csv
```

### `prepare_alphagenome_supported_benchmark.py`

Converts ontology support into exact AlphaGenome RNA-seq track names, for
example `UBERON:0001159 total RNA-seq`, and writes track-to-qnorm mappings.

```bash
python downstream_tasks/expression_prediction/alphagenome_benchmark/scripts/prepare_alphagenome_supported_benchmark.py \
  --support-summary data/qnorm_alphagenome_support_summary.csv \
  --alphagenome-rna-metadata data/alphagenome_human_rna_seq_metadata.csv \
  --mapping-csv data/alphagenome_track_to_qnorm_ids.csv \
  --mapping-py data/alphagenome_track_to_qnorm_ids.py \
  --excluded-csv data/alphagenome_ontology_only_excluded.csv
```

## 6. Files To Commit

Commit scripts, configs, small mapping files, and this README.

Do not commit:

- API keys.
- `/scratch/...` outputs.
- large matrices such as `data/borzoi_all_ids_qnorm_matrix.csv`.
- metadata caches such as `data/encode_metadata_cache.json`.

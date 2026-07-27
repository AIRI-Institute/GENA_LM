# GENA_LM Expression Benchmark

This folder contains the scripts used to run GENA_LM expression inference for
the human `valid` and `test` gene sets.

It is the GENA_LM part of the joint GENA_LM vs AlphaGenome benchmark branch.

## What The Model Predicts

For every gene and every cell-line JSON description, the script predicts one
expression value:

```text
gene sequence around TSS + cell-line JSON description -> scalar expression prediction
```

The output matrix format is:

```text
gene_id,ENCFF035CWS,ENCFF083EOC,...
ENSG00000232604.1,0.322,0.005,...
```

Rows are genes. Columns are cell IDs.

## Files In This Folder

```text
scripts/get_json_folder.py
scripts/run_gena_lm_model_inference.sh
scripts/run_full_model_inference.sh
scripts/run_all_gena_lm_models.sh
scripts/score_ct_specificity.py
```

The main Python inference runner is stored in:

```text
downstream_tasks/expression_prediction/inference_example/run_polina_batch_inference.py
```

It uses:

```text
downstream_tasks/expression_prediction/inference_example/inference_input_utils.py
```

## Required Input Layout On Anogena

The scripts assume this working folder:

```text
/home/jovyan/dpanc/benchmarking/GENA_LM
```

Required folders/files:

```text
/home/jovyan/dpanc/benchmarking/data/human.valid.forward.csv
/home/jovyan/dpanc/benchmarking/data/human.valid.reverse.csv
/home/jovyan/dpanc/benchmarking/data/human.test.forward.csv
/home/jovyan/dpanc/benchmarking/data/human.test.reverse.csv
/home/jovyan/dpanc/benchmarking/data/hg38.fna

/home/jovyan/dpanc/benchmarking/GENA_LM/models/<model_name>/...
/home/jovyan/dpanc/benchmarking/GENA_LM/inference_inputs/*.csv
/home/jovyan/dpanc/benchmarking/GENA_LM/metadata_borzoi_all/*.json
/home/jovyan/dpanc/benchmarking/GENA_LM/json_runs/json_14/*.json
/home/jovyan/dpanc/benchmarking/GENA_LM/json_runs/json_812/*.json
```

The repo path is expected to be:

```text
/home/jovyan/dpanc/GENA_LM/GENA_LM_expression_branch
```

If needed, override paths:

```bash
export TASK_ROOT=/path/to/benchmarking/GENA_LM
export GENA_HOME=/path/to/GENA_LM/repo
export DATA_ROOT=/path/to/benchmarking/data
```

## Create JSON Folders

Create a folder with 14 cell-line JSONs:

```bash
python downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/get_json_folder.py \
  --map /home/jovyan/dpanc/benchmarking/GENA_LM/inference_inputs/Expression_dataset_v1_csv_file_mappings_qnorm_14_fixed.csv \
  --task-root /home/jovyan/dpanc/benchmarking/GENA_LM \
  --out-dir /home/jovyan/dpanc/benchmarking/GENA_LM/json_runs/json_14
```

Create a folder with 812 cell-line JSONs:

```bash
python downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/get_json_folder.py \
  --map /home/jovyan/dpanc/benchmarking/GENA_LM/inference_inputs/file_mappings_borzoi_all_GRCh38_only_with_description.csv \
  --task-root /home/jovyan/dpanc/benchmarking/GENA_LM \
  --out-dir /home/jovyan/dpanc/benchmarking/GENA_LM/json_runs/json_812
```

Check:

```bash
find /home/jovyan/dpanc/benchmarking/GENA_LM/json_runs/json_14 -name "*.json" | wc -l
find /home/jovyan/dpanc/benchmarking/GENA_LM/json_runs/json_812 -name "*.json" | wc -l
```

Expected:

```text
14
812
```

## Run One Model

Activate the environment:

```bash
source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate api
```

Run one checkpoint:

```bash
cd /home/jovyan/dpanc/GENA_LM/GENA_LM_expression_branch

bash downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/run_gena_lm_model_inference.sh \
  glioma valid json_812 4 256
```

Arguments:

```text
MODEL_NAME SPLIT CELL_SET GPU_ID BATCH_SIZE
```

Allowed models:

```text
full_model
all_datasets_2
decoder
dev_loss
glioma
len_2048
modernbert_large
```

Allowed splits:

```text
valid
test
```

Allowed cell sets:

```text
json_14
json_812
```

The `len_2048` model automatically uses `--dna-max-seq-len 2048`. All other
models use 1024 DNA tokens.

## Run Full Model Only

Example:

```bash
cd /home/jovyan/dpanc/GENA_LM/GENA_LM_expression_branch

nohup bash downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/run_full_model_inference.sh \
  valid json_812 3 256 \
  > /home/jovyan/dpanc/benchmarking/GENA_LM/outputs/logs/full_model_valid_json812.nohup.log 2>&1 &
```

For test:

```bash
nohup bash downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/run_full_model_inference.sh \
  test json_812 3 256 \
  > /home/jovyan/dpanc/benchmarking/GENA_LM/outputs/logs/full_model_test_json812.nohup.log 2>&1 &
```

## Run All Six Benchmark Checkpoints

This runs valid and test for:

```text
all_datasets_2
decoder
dev_loss
glioma
len_2048
modernbert_large
```

Example for 812 cell lines:

```bash
cd /home/jovyan/dpanc/GENA_LM/GENA_LM_expression_branch

nohup bash downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/run_all_gena_lm_models.sh \
  json_812 4 256 \
  > /home/jovyan/dpanc/benchmarking/GENA_LM/outputs/logs/all_models_json812.nohup.log 2>&1 &
```

Example for 14 cell lines:

```bash
nohup bash downstream_tasks/expression_prediction/gena_lm_benchmark/scripts/run_all_gena_lm_models.sh \
  json_14 4 256 \
  > /home/jovyan/dpanc/benchmarking/GENA_LM/outputs/logs/all_models_json14.nohup.log 2>&1 &
```

## Outputs

Predictions are written to:

```text
/home/jovyan/dpanc/benchmarking/GENA_LM/predictions_results/<model_name>/gena_lm_<split>_<cell_set>_predictions.csv
```

Example:

```text
/home/jovyan/dpanc/benchmarking/GENA_LM/predictions_results/glioma/gena_lm_valid_json812_predictions.csv
```

Logs are written to:

```text
/home/jovyan/dpanc/benchmarking/GENA_LM/outputs/logs
```

## What Was Changed For Git

The original working scripts were kept as the source, but the git version was
made less ambiguous:

- `CELL_SET` is now explicit: `json_14` or `json_812`.
- one generic script runs one model/split/cell-set combination.
- one wrapper runs the six benchmark checkpoints for both `valid` and `test`.
- full model has a tiny separate wrapper for quick single-checkpoint runs.
- model files, output CSVs, logs, and caches are not committed.

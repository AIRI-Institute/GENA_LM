# GENA_LM Expression Benchmark

This folder contains the scripts used to run GENA_LM expression inference for
the human `valid` and `test` gene sets.

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



## Required input

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

/home/jovyan/dpanc/benchmarking/GENA_LM/models/<model_name>/<model_name>pytorch_model.bin
/home/jovyan/dpanc/benchmarking/GENA_LM/inference_inputs/*.csv
/home/jovyan/dpanc/benchmarking/GENA_LM/metadata_borzoi_all/*.json
/home/jovyan/dpanc/benchmarking/GENA_LM/json_runs/json_14/*.json
/home/jovyan/dpanc/benchmarking/GENA_LM/json_runs/json_812/*.json
```



## Download json files

The two mapping files used for benchmark runs were:

```text
/home/jovyan/dpanc/benchmarking/GENA_LM/inference_inputs/Expression_dataset_v1_csv_file_mappings_qnorm_14_fixed.csv
/home/jovyan/dpanc/benchmarking/GENA_LM/inference_inputs/file_mappings_borzoi_all_GRCh38_only_with_description.csv
```

We used `scripts/get_json_folder.py` to create run-specific JSON folders. The
script reads the mapping CSV, takes the `id` and `metadata` columns, and creates
symlinks named like `ENCFF035CWS.json`. So:

- `json_runs/json_14` contains 14 symlinks for the small benchmark.
- `json_runs/json_812` contains 812 symlinks for the full benchmark.

If the metadata JSON folder is missing, restore it from project storage/S3 first. The bucket path we used for expression metadata/data

```text
s3://genalm/expr/data/tpm/
```

With configured S3 credentials

```bash
aws s3 cp \
  s3://genalm/expr/data/tpm/ \
  /home/jovyan/dpanc/benchmarking/GENA_LM/ \
  --recursive \
  --profile airi \
  --endpoint-url https://s3.cloud.ru \
  --region ru-central-1
```

After download, check where the JSON files are and make/link this folder:

```bash
find /home/jovyan/dpanc/benchmarking/GENA_LM -name "ENCFF*.json" | head
mkdir -p /home/jovyan/dpanc/benchmarking/GENA_LM/metadata_borzoi_all
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



## Run One Model

Activate the environment

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
dev_loss
glioma
len_2048
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

The `len_2048` model automatically uses `--dna-max-seq-len 2048`. All other models use 1024 DNA tokens

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

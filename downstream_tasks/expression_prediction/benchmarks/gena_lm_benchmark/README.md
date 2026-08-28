# GENA-LM expression benchmark

Run GENA-LM expression inference for human or mouse `valid` and `test` genes.
Commands can be launched from any directory after cloning the repository.

## Ground-truth comparison notebooks

The `notebooks/` folder contains GENA-LM prediction comparisons with ground
truth:

- `gena_14_celllines_compare.ipynb`
- `gena_lm_gt_compare.ipynb`
- `gena_lm_gt_compare_loop.ipynb`
- `gena_lm_gt_compare_mm10.ipynb`

## Environment and paths

Activate an environment containing the project dependencies, then set the data
location if it is outside the clone:

```bash
cd /path/to/GENA_LM
conda activate api

export GENA_HOME="$PWD"
export TASK_ROOT="${TASK_ROOT:-$GENA_HOME}"
export DATA_ROOT="${DATA_ROOT:-$GENA_HOME/data}"
```

The default layout is:

```text
<repository>/
├── data/
│   ├── human.valid.forward.csv
│   ├── human.valid.reverse.csv
│   ├── human.test.forward.csv
│   ├── human.test.reverse.csv
│   ├── mouse.valid.forward.csv
│   ├── mouse.valid.reverse.csv
│   ├── mouse.test.forward.csv
│   ├── mouse.test.reverse.csv
│   ├── hg38.fna
│   └── mm10.fa
├── models/<model_name>/
├── json_runs/
├── metadata_borzoi_all/
├── predictions_results/
├── predictions_results_mm10/
└── downstream_tasks/
```

For external storage, set `TASK_ROOT`, `DATA_ROOT`, or `PYTHON_BIN`.
Explicit command-line paths take precedence in the Python runners.

## Prepare metadata JSON folders

`scripts/get_json_folder.py` reads the `id` and `metadata` columns of a mapping
CSV and creates one JSON symlink per sample:

```bash
python downstream_tasks/expression_prediction/benchmarks/gena_lm_benchmark/scripts/get_json_folder.py \
  --map "$TASK_ROOT/inference_inputs/mapping.csv" \
  --task-root "$TASK_ROOT" \
  --out-dir "$TASK_ROOT/json_runs/json_14"
```

Use `--copy` when symlinks are unsuitable. Metadata used in the original
benchmark is stored under `s3://genalm/expr/data/tpm/` and requires separate S3
credentials.

## Human inference

Run one model:

```bash
bash downstream_tasks/expression_prediction/benchmarks/gena_lm_benchmark/scripts/run_gena_lm_model_inference.sh \
  glioma valid json_812 4 256
```

Arguments are:

```text
MODEL_NAME SPLIT CELL_SET GPU_ID [BATCH_SIZE]
```

Supported models:

```text
ATAC
all_datasets
all_datasets2
all_datasets_2
dev_loss
full_model
glioma
len_2048
mult_loss
xlarge
```

Each checkpoint uses the experiment YAML in the same model directory.
`len_2048` uses 2048 DNA tokens; all other launch entries use 1024.

Run every model for both splits:

```bash
nohup bash downstream_tasks/expression_prediction/benchmarks/gena_lm_benchmark/scripts/run_all_gena_lm_models.sh \
  json_812 4 256 \
  > "$TASK_ROOT/outputs/logs/all_models_json812.log" 2>&1 &
```

## Mouse inference

The mouse runner defaults to `mouse.<split>.forward.csv`,
`mouse.<split>.reverse.csv`, and `mm10.fa` under `DATA_ROOT`.

```bash
MODEL_DIR="$TASK_ROOT/models/glioma"

CUDA_VISIBLE_DEVICES=4 python -u \
  downstream_tasks/expression_prediction/inference_example/run_polina_batch_inference_mouse.py \
  --experiment-config "$MODEL_DIR/glioma.yaml" \
  --checkpoint "$MODEL_DIR/glioma_pytorch_model.bin" \
  --json-dir "$TASK_ROOT/json_runs/json_243_mm10" \
  --split valid \
  --device cuda:0 \
  --batch-size 256 \
  --output "$TASK_ROOT/predictions_results_mm10/glioma/gena_lm_valid_json243_mm10_predictions.csv"
```

Change the checkpoint, colocated experiment config, split, and output together
for another run. Add `--dna-max-seq-len 2048` for `len_2048`.

## Outputs

Human launchers write predictions to:

```text
$TASK_ROOT/predictions_results/<model_name>/gena_lm_<split>_<cell_set>_predictions.csv
```

Logs are stored under `$TASK_ROOT/outputs/logs`. Keep models,
reference genomes, metadata, predictions, and logs outside Git.

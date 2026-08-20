# AlphaGenome expression benchmark

Run AlphaGenome RNA-seq predictions for human or mouse genes and compare them
with qnorm expression. The JSON workflow supports checkpoints and is preferred;
the older script keeps the original 14-cell-type mapping in Python.

## Ground-truth comparison notebooks

The `notebooks/` folder contains AlphaGenome prediction comparisons with ground
truth:

- `gena_ag_gt_ontology_compare.ipynb`: human comparison
- `gena_ag_gt_ontology_compare_mm10.ipynb`: mouse comparison

## Setup

From the cloned repository, activate the environment and select the external
data/output root:

```bash
cd /path/to/GENA_LM
export GENA_HOME="$PWD"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$GENA_HOME}"
export DATA_ROOT="${DATA_ROOT:-$GENA_HOME/data}"

conda create -n alphagenome python=3.11 pip -y
conda activate alphagenome
python -m pip install alphagenome==0.7.0 biopython pyyaml tqdm requests

cd downstream_tasks/expression_prediction/benchmarks/alphagenome_benchmark
```

Configs use `${DATA_ROOT}` for inputs and `${BENCHMARK_ROOT}` for outputs; the
scripts expand both at runtime. They default to `data/` and the repository root.

Version `0.7.0` provides the strand-aware scoring API used by the benchmark.
Keep credentials outside Git:

```bash
export ALPHAGENOME_API_KEY="your_api_key_here"
```



## Run predictions

The JSON runner reads track-to-qnorm mappings from
`ALPHAGENOME_ONTOLOGY_JSON`.

### Mouse

Check the input and output paths in
`scripts/config_alphagenome_supported_mouse.yaml`. In
`scripts/Alphagenome_prediction_benchmarking_json.py`, also set the mouse
organism in `model.score_interval()`:

```python
organism=dna_client.Organism.MUS_MUSCULUS
```

Choose the input interval in `make_interval()`:

```python
# 1 Mb
interval = interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

# or 16 Kb
interval = interval.resize(dna_client.SEQUENCE_LENGTH_16KB)
```

Use a matching `GeneMaskScorer` width (`200_001` for 1 Mb or `10_001` for
16 Kb), and keep the interval size consistent with output and log names. Then
run:

```bash
export ALPHAGENOME_ONTOLOGY_JSON="$PWD/data/alphagenome_track_to_qnorm_id_mouse_first.json"
mkdir -p logs

nohup env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  python3 -u scripts/Alphagenome_prediction_benchmarking_json.py \
  --config scripts/config_alphagenome_supported_mouse.yaml \
  > logs/alphagenome_mouse_1Mb.log 2>&1 &

tail -f logs/alphagenome_mouse_1Mb.log
```

Proxy variables are removed because an unavailable local proxy can block the
AlphaGenome gRPC client at `dna_client.create()`.

### Human

```bash
export ALPHAGENOME_ONTOLOGY_JSON="$PWD/data/alphagenome_track_to_qnorm_id_first.json"
sbatch scripts/run_alphagenome_json.sbatch
```

The human config is `scripts/config_alphagenome_supported.yaml`. The original
14-cell-type workflow uses `scripts/Alphagenome_prediction_benchmarking.py`,
`scripts/config.yaml`, and `scripts/run_alphagenome_supported.sbatch`.

## Configuration and outputs

Before running, check:

- `output_dir` and `checkpoint_path`
- `splits` (`test`, `valid`)
- `prediction_functions` (normally `intervals`)
- all gene interval, truth, and reference-genome paths

`checkpoint_every` controls partial saves. Rerunning the same command resumes
from the checkpoint.

Expected files:

```text
alphagenome_supported_predictions_test_intervals.csv
alphagenome_supported_predictions_valid_intervals.csv
Alphagenome_benchmark_summary.csv
checkpoint_test_intervals.csv
checkpoint_valid_intervals.csv
```

Prediction tables contain one row per gene and one column per mapped RNA-seq
track. An all-`NaN` column means AlphaGenome did not return that requested track.

## Build mouse ontology mappings

The following commands assume the current directory is this benchmark folder.

1. Export AlphaGenome mouse RNA-seq metadata. In
  `dump_alphagenome_rna_metadata.py`, use
   `dna_client.Organism.MUS_MUSCULUS`.

```bash
python scripts/dump_alphagenome_rna_metadata.py \
  --out data/alphagenome_mouse_rna_seq_metadata.csv
```

1. Map qnorm experiments to ENCODE biosample ontologies.

```bash
python scripts/build_alphagenome_ontology_map.py \
  --qnorm-map /path/to/true_mouse_all_genes_qnorm_samples_by_genes.WITH_TEST.csv \
  --out data/qnorm_alphagenome_ontology_map_mouse.csv \
  --cache data/encode_metadata_cache_mouse.json
```

1. Match those ontologies to AlphaGenome tracks.

```bash
python scripts/make_experiment_ontology_dict.py \
  --ontology-map data/qnorm_alphagenome_ontology_map_mouse.csv \
  --alphagenome-rna-metadata data/alphagenome_mouse_rna_seq_metadata.csv \
  --dict-py data/experiment_ontology_dict_mouse.py \
  --summary-csv data/qnorm_alphagenome_support_summary_mouse.csv
```

1. Create the JSON mappings with
  `scripts/prepare_json_for_running.ipynb`:

- `*_first.json`: one representative qnorm ID per AlphaGenome track; used to
name prediction columns.
- `*_all.json`: all qnorm IDs per track; used to aggregate ground truth.

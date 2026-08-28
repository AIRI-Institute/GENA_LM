# Guide: data preparation and model fine-tuning

Instructions for `GENA_LM/downstream_tasks/expression_prediction`.

The order of work is always the same:

```
data  → file_mappings + metadata      (section 2)
      → qnorm                         (section 3)
      → experiment config             (section 4)
      → cache precompute              (section 5)
      → training                      (section 6)
```

If the data is already prepared and you only need to fine-tune the model, go to sections 4–6.

---

## 1. Environment

### 1.1. Variables

Everything hinges on `GENALM_HOME`, which is expected to contain a `GENA_LM` folder:

```
$GENALM_HOME/
├── GENA_LM/                # repository, expression branch
├── models/                 # backbones and decoders
├── runs/                   # checkpoints are written here
├── soft/t5-experiments/    # trainer wrapper
```

```bash
export GENALM_HOME=/path/to/your/workdir
export EP="$GENALM_HOME/GENA_LM/downstream_tasks/expression_prediction"
```

In the configs the same thing is written as `${oc.env:GENALM_HOME}` → `${HOME_PATH}`.

### 1.2. Installing the environment

```bash
conda create -n expression python=3.11
conda activate expression
pip install -r "$EP/requirements.txt"

mkdir -p "$GENALM_HOME/soft" && cd "$GENALM_HOME/soft"
git clone --branch feat/trainer_with_accelerate https://github.com/yurakuratov/t5-experiments.git
cd t5-experiments
sed -i 's/requirements-lm-tools.txt/requirements-lm-tools-accel.txt/' setup.py
python -m pip install -e .
```

Additionally: `pip install qnorm` for normalizing tabular data (3.1); `pyBigWig` and `matplotlib` for bigWig (3.2).

### 1.3. Layout of the main folder

```
expression_prediction/
├── configs/                       # experiments (*.yaml); configs/legacy/ is an archive
├── datasets/
│   ├── data/                      # <DATASET>/, genomes/, file_mappings/, tpm/, bw/
│   ├── src/                       # data preparation scripts and tabular qnorm
│   └── README.md                  # where to download ready-made data from
├── intervals/                     # intervals + dataset_hash.*.h5 caches
├── descriptions/                  # description embedding caches
├── src/precompute_datasets.py     # cache warm-up
├── expression_dataset_final.py    # dataset classes
├── expression_model_final.py      # ExpressionCounts model
├── run_expression_finetuning_final.py
└── finetune_expression.sh         # training launcher
```

---

## 2. Preparing datasets

### 2.1. Dataset folder

```
$EP/datasets/data/<DATASET_NAME>/
├── file_mappings.csv          # dataset index
├── metadata/<sample>.json     # one per track
└── tpm/<sample>.csv           # or cpm/ — 1×N tables; for coverage, bigWig files instead
```

After normalization, `<sample>_qnorm.csv` and `file_mappings_qnorm.csv` appear alongside them — the originals are left untouched.

### 2.2. `file_mappings.csv`

| Column | Description |
|---|---|
| `id` | unique track ID; usually an md5 of the metadata contents |
| `metadata` | path to the metadata json |
| `assay` | CAGE, ATAC, RNAseq, scRNA, snRNA… |
| `genome` | `hg38`, `mm10`, `GCF_...` |
| `processed` | who prepared the data |
| `dataset_description` | dataset description; metric names are built from it (4.5) |
| tabular: `csv` / `tsv` / `CPM` / `tpm` | the column name is specified in the config via the `tpm:` field |
| `forward_bw`, `reverse_bw` | for coverage |

```csv
id,tpm,metadata,assay,genome,dataset_description,processed
Aspc_r1,./tpm/Aspc_r1.csv,./metadata/Aspc_r1.json,RNAseq,hg38,homebrew,Galya
```

**All paths inside are relative to `file_mappings.csv` itself.** The folder can be moved as a whole without editing anything.

Do not mix tracks of different types in one dataset (bw only / tpm only / both) — each type gets its own config block. When merging your file_mappings with someone else's, use `combine_filemapping()` from `datasets/src/utils.py`: it merges by `id` and does not overwrite other people's rows.

### 2.3. `metadata/<sample>.json`

Required:

- **cell type** or **tissue type** — `cell_type` / `tissue`;
- **assay**;
- **`genome`** — must match the `genome` column in file_mappings.

`id` is not required. For `norm_bw: True` you additionally need `forward_total_coverage` and `reverse_total_coverage` (3.3).

Field names are not standardized — what matters is the content, because the track description used for embeddings is assembled from the metadata. The more specific the cell type and conditions, the better the model tells tracks apart: given the same input sequence, the description is the only thing that distinguishes them.

```json
{"id": "0a36d0933f0d32fdfd074a86a576af22", "genome": "hg38", "cell_type": "Type II myofibres",
 "assay": "scRNA", "tissue": "muscle",
 "description": "Count matrix for celltype psudobulks. Genes are Gene Symbols and values are in CPM"}
```

The schema can be entirely different — in `RNAseq_homebrew` there is no `id` field at all, and the cell type lives in `Characteristics [Cell type]`, yet it is a working dataset.


### 2.4. Genome and intervals

```
$EP/datasets/data/genomes/<genome>/
├── <genome>.fa                 # required
├── <genome>.chrom.sizes        # recommended
└── <genome>_genes.csv          # annotation: same columns as the intervals, plus split
```

```
$EP/intervals/<genome>.<split>.<strand>.csv
```

`<split>`: `train` | `valid` | `test`, `<strand>`: `forward` | `reverse`. **The separator is a tab.** Required columns: `gene_id`, `strand`, `chromosome`, `TSS`, `TES`; `gene_name` is desirable. For multi-species the layout is nested: `intervals/<ASSEMBLY>/<ASSEMBLY>.train.forward.tsv`.

---

## 3. Qnorm

Different experiments arrive on different scales: different sequencing depth, different protocol, different lab. Without aligning them the model learns to tell protocols apart rather than biology. Qnorm brings all tracks to a common distribution.

The scripts live in `$EP/datasets/src/` and are run **from that directory** — except the per-chromosome bigWig variant, which lives elsewhere (3.2.1).

What to run:

| Target | Situation | Script |
|---|---|---|
| TPM/CPM tables | single assembly, annotation and intervals available | `qnorm_tpm_data.py` (3.1.1) |
| TPM/CPM tables | many assemblies, no common annotation | `qref_norm.py` (3.1.2) |
| TPM/CPM tables | samples added to an already normalized set | `apply_ref.py` (3.1.3) |
| bigWig | one genome | `qnorm_bigwig_fixed.py` → `qnorm_bigwigs()` (3.2.1) |
| bigWig | several assemblies, whole set in one go | `qnorm_bw_global_reference.py` (3.2.2) |
| bigWig | several assemblies, in parts or with later additions | `qnorm_bw_build_reference.py` + `qnorm_bw_apply_reference.py` (3.2.3) |

There is a single principle behind the choice: either the distribution is derived from the data itself — and then it shifts whenever samples are added — or it is fixed once into a reference file and afterwards applied to new data without moving the old. **For growing sets and for multi-species, always the second path.**

The originals are never overwritten: normalized data is written next to them with a suffix.

The `norm_bw` flag (3.3) stands apart: it is not quantile normalization but a division of the signal by a single coefficient — the track's total coverage taken from the metadata. It replaces nothing in the table and combines with any row in it.

### 3.1. TPM/CPM tables

#### 3.1.1. `qnorm_tpm_data.py` — single assembly

**When:** all tracks belong to one assembly, and `<genome>_genes.csv` plus intervals are available.

```bash
cd "$EP/datasets/src"
python qnorm_tpm_data.py \
  --filemappings_paths ../data/<DATASET_NAME>/file_mappings.csv \
  --save_path ../data/tpm/<DATASET_NAME>_ \
  --intervals_prefix ../../intervals/hg38.
```

`--save_path` is a **prefix**, not a folder: file names are appended to it. `--intervals_prefix` is a prefix too, with `{split}.{strand}.csv` appended.

**What it does:** collects the tables of all tracks, merges them by `<genome>_genes.csv`, labels genes as train/valid/test using the intervals, normalizes, then drops genes with `max(TPM) < 2` from train — but only if they account for no more than 20% (otherwise it logs that filtering was skipped, which is a reason to check the data scale: is this really TPM?).

**Output:**

```
<save_path>tpms.csv            # genes × tracks matrix, raw
<save_path>qnorm_tpms.csv      # the same after normalization
<sample>_qnorm.csv             # next to each source file
file_mappings_qnorm.csv        # next to the original, with updated paths
```

That last file is what later goes into `targets_path` in the config.

**Important:**

- all file_mappings in one run must belong to **one assembly**: there is an `assert` on a single `genome` inside. Human and mouse are normalized in separate runs;
- `--intervals_prefix` must be from the same assembly, otherwise the merge with the intervals comes out empty;
- `gene_id` values in the tables must match `<genome>_genes.csv` — on a mismatch the script fails with `Less than 1000 genes in <file>`;
- the tabular column must be named `csv`, `tsv` or `CPM`. This script does not see a `tpm` column — use `qref_norm.py` for that;
- `--filemappings_paths` is declared as `nargs="+"`: list all paths **after a single flag**. A flag given twice keeps only the last value, and part of the data silently never reaches normalization.

Requires `pip install qnorm`. Other arguments: `--ncpus` (8), `--num_cell_types` (for debugging), `--selected_targets_path`, `--pseudo_count`.

#### 3.1.2. `qref_norm.py` — multi-species

**When:** many assemblies, no common annotation.

```bash
cd "$EP/datasets/src"
python qref_norm.py ../data/<DATASET_NAME>/file_mappings.csv --tpm-col tpm --suffix _qref
```

**What it does:** builds a reference distribution as the median of the quantile functions across all samples and maps every sample onto it. Zeros stay zeros.

**Output:**

```
file_mappings_qref.csv         # updated paths
file_mappings_qref_ref.npy     # the reference — keep it, it is needed for 3.1.3
file_mappings_qref_ref.csv     # the same in readable form
<sample>_qref.csv              # one per sample
```

#### 3.1.3. `apply_ref.py` — add samples to an existing set

**When:** the set has grown and the already computed distribution must not shift.

```bash
cd "$EP/datasets/src"
python apply_ref.py <REF_NPY> <FILE_MAPPINGS_CSV> --tpm-col tpm --suffix _qref
```

**What it does:** takes the ready `_ref.npy` from 3.1.2 and applies it to the new samples. Positive values are mapped onto the reference, zeros stay zeros, NaN/Inf are left alone.

**Output:** `<sample>_qref.csv` and an updated file_mappings.

Re-running `qref_norm.py` over the extended set is not a substitute here: it would shift the distribution and put the new data out of step with an already trained model.

### 3.2. bigWig

There are two different approaches here, and the choice is determined by the number of assemblies.

| | one genome | several assemblies |
|---|---|---|
| Reference | **one per chromosome** | **a single shared one** for everything |
| Tool | `qnorm_bigwig_fixed.py` (3.2.1) | `qnorm_bw_*_reference.py` (3.2.2–3.2.3) |

The logic is simple: in samples from one assembly the chromosomes correspond to each other, so the distribution can be aligned separately for each — which is more precise. Across species chromosomes are not comparable, leaving one distribution for the whole genome.

Common to both: the **input** is a CSV/TSV table with a column of bigWig paths, the **output** is new `.qnorm.bw` files next to the originals. `pyBigWig` is required; `matplotlib` for the quantile-curve plots.

Three things everyone trips over:

- **paths are taken from the table as-is**, without being resolved relative to it — unlike `file_mappings.csv`. Absolute paths are required, otherwise the files will not be found;
- there are two signal columns, `forward_bw` and `reverse_bw`, which means **two runs**, one per strand;
- the scripts do not edit `file_mappings.csv` — copy the new paths there yourself.

#### 3.2.1. One genome: per-chromosome reference

**When:** all bigWigs belong to a single assembly.

The `qnorm_bigwigs()` function. **For each chromosome separately** it builds a reference as the per-quantile median across all samples and maps every file onto it. Zeros form their own tie group and, with `tie_stat="min"` (the default), stay zeros.

There is no CLI, and no import either: in the notebooks the whole function body is simply pasted into cells, and `qnorm_bigwig_fixed.py` is an exported copy of that same code. So you either run the definition cells, or put the `.py` next to your notebook and import it yourself.

An actual call from `qnorm_bw_process.ipynb` (mouse mm39):

```python
out_df = qnorm_bigwigs(
    df_bw,
    bw_col="bw",
    out_suffix=".qnorm_final.bw",
    grid_size=200001,
    round_decimals=6,
    chrom_filter=None,
    skip_existing=True,
    tie_stat="min",
    show_progress=True,
    zero_clip_eps=1e-12,
    blacklist_bed=".../datasets/data/ATACseq_Egor/mm39.blacklist.bed",
    save_ref_to_bed_dir=True,
    log_level=logging.INFO,
)
```

For human, `qnorm_bw.ipynb` does the same with `hg38_blacklist.v2.bed` and an explicit `chrom_filter=["chr1", ..., "chrM"]`.

What to watch for in the parameters:

- `blacklist_bed` — those regions are excluded from the statistics and **not written** to the output, they stay as holes. In the actual runs it is always set, one per genome;
- `grid_size=200001` — the working runs use exactly this, not the default `20001`;
- `save_ref_to_bed_dir=True` — the reference and the plots go into the folder holding the BED;
- `chrom_filter` — either `None` or an explicit list of chromosomes;
- `upper_tail_p` (0.999999) — winsorization of the upper tail before the reference is built; `plot_qcurves` — quantile curves per chromosome.

> The files live outside the repository, on the `aspeedok` host under `/home/jovyan/shares/SR003.nfs2/aspeedok/notebooks/`: `qnorm_bigwig_fixed.py` itself and the notebooks `qnorm_bw.ipynb` (human), `qnorm_bw_process.ipynb` (mouse), `qnorm_multispecies.ipynb` (multi-species).

#### 3.2.2. Several assemblies: one shared reference

There are three scripts, related as follows:

```
$EP/datasets/src/
├── qnorm_bw_global_reference.py    # shared core + "build and apply in one run" mode
├── qnorm_bw_build_reference.py     # build the reference only
└── qnorm_bw_apply_reference.py     # apply an existing one only
```

`qnorm_bw_global_reference.py` is build + apply together, and it also holds all the working code that the other two import. Keep the files in one folder: on their own, build and apply will not run.

| | `global_reference` | `build` + `apply` |
|---|---|---|
| Reference is built from | the input files | the input files |
| Applied to | the **same** files | any set |
| Where the reference is saved | `qnorm_ref_global.tsv.gz` next to `--input`, name is fixed | the path you give in `--out-ref` |

Hence the rule: **you must not run `global_reference` separately per species** — each species would get its own reference and its own scale. One reference for all is needed: `build` once, `apply` per set.

The project's ready-made global reference (200001 points): `$GENALM_HOME/src/qnorm_ref_global_ref.tsv.gz`.

Everything at once, when the set is not going to grow:

```bash
cd "$EP/datasets/src"
python qnorm_bw_global_reference.py \
  --input /abs/path/bw_table.csv --bw-col forward_bw \
  --out-table /abs/path/bw_table.forward.qnorm.csv \
  --ref-chrom-count 5 --chrom-jobs 16
```

The reference is saved along the way as `qnorm_ref_global.tsv.gz` in the `--input` folder — if you later need to bring new files onto the same scale, feed it to `qnorm_bw_apply_reference.py`.

#### 3.2.3. In parts: reference separately, application separately

**When:** the set is not normalized in one go — data arrives in batches, or it is more convenient to process one species/batch at a time. The reference is built **once over the whole set** and then applied to each part; that way all parts end up on one scale, and adding a new one does not shift those already normalized.

The typical case is multi-species: the 103 species in `ATACseq_Egor_Multispecies` were normalized exactly this way. But it is not about species as such: a single assembly split into batches calls for the same approach.

Step 1, build the reference:

```bash
python qnorm_bw_build_reference.py \
  --input /abs/path/all_species_bw.csv --bw-col forward_bw \
  --out-ref /abs/path/ref.tsv.gz \
  --ref-chrom-count 5 --ref-row-step 10 \
  --plot-qcurves --plots-dir /abs/path/qc_plots
```

The output is a TSV.GZ with columns `q` and `x_ref`.

Step 2, apply it to each set:

```bash
python qnorm_bw_apply_reference.py \
  --input /abs/path/<Species>_bw.csv --ref /abs/path/ref.tsv.gz \
  --bw-col forward_bw \
  --out-table /abs/path/<Species>_bw.forward.qnorm.csv \
  --chrom-jobs 16
```

The remaining flags (`--grid-size`, `--upper-tail-p`, `--tie-stat`, `--skip-existing`, `--chroms`, `--chunk-size-bp`) are in `--help`; the defaults work.

### 3.3. `norm_bw` — division by a coefficient, not qnorm

This is not an alternative to 3.2 but a different operation, and the two do not conflict. Qnorm remaps values onto a common distribution and does so in advance, into separate files. `norm_bw` is a single coefficient per track, applied at read time:

```yaml
train_dataset_<name>:
  bw: bw
  norm_bw: True
```

The dataset divides the signal by the track's total coverage. It takes the coefficient from **each track's metadata json** — the fields `forward_total_coverage` / `reverse_total_coverage`, which `datasets/src/bam2bw.py` fills in during BAM → bigWig conversion.

What it gives you: it removes the difference in library size. What it does not: track distributions do not become identical — dividing by a constant does not change the shape of a distribution. That is why the flag can be applied both to already normalized `.qnorm.bw` files and to raw ones.

**Important:**

- the fields are read from the json, not from the identically named file_mappings columns; if they are missing there, `Missing normalization factor` goes to the log and the track stays unnormalized;
- the flag is part of the signal cache hash — toggling it requires a recompute (section 5);
- for `MethylationDataset` it is forbidden by an explicit `assert`: methylation is already expressed as a 0..1 fraction.

---

## 4. Config

Hydra/OmegaConf YAML in `$EP/configs/`. Use the current `final_*.yaml` files as a starting point; everything in `configs/legacy/` is an archive and should not be copied from.

### 4.1. Header

```yaml
TASK_NAME: expression/my_experiment       # determines where checkpoints go
HOME_PATH: ${oc.env:GENALM_HOME}
MODEL_PATH_ROOT: ${HOME_PATH}/runs/${TASK_NAME}
```

The run's output ends up in `$GENALM_HOME/runs/expression/my_experiment/<timestamp>/`. **Change `TASK_NAME` for every experiment** — otherwise runs will pile up in one folder.

### 4.2. Model

```yaml
model_kwargs:
  _target_: builtins.dict
  hf: True
  hf_model_name: "${HOME_PATH}/models/<backbone>"          # or a model name from the HF Hub
  hf_model_name_decoder: "${HOME_PATH}/models/<decoder>"
  desc_model_name: "Qwen/Qwen3-Embedding-0.6B"
```

Check that the backbone and the decoder are present in `$GENALM_HOME/models/` before launching — otherwise you will crash only after the caches have been warmed up.

### 4.3. Dataset blocks

#### How they are named

Each dataset is a separate top-level key, and the prefix decides everything:

- `train_dataset_<any suffix>` → goes into training;
- `valid_dataset_<any suffix>` → goes into validation.

The suffix after the prefix affects nothing, it is there for readability. There can be any number of blocks of each kind — all `train_dataset*` are combined into one training set, all `valid_dataset*` into the validation set.

#### What every block must have

| Field | Meaning |
|---|---|
| `_target_` | dataset class, see below |
| `targets_path` | the **normalized** file_mappings (`*_qnorm.csv` / `*_qref.csv`), not the original |
| `genome` | path to the `.fa` |
| `forward_intervals_path`, `reverse_intervals_path` | intervals for the relevant assembly; `train` in train blocks, `valid` in valid blocks |
| `gen_max_seq_len` | input length in tokens; must match `input_seq_len` |
| `num_before` | how many tokens to take to the left of the TSS |

#### How tabular and coverage blocks differ

This is the main difference between blocks — which field defines the target:

| | tabular (TPM/CPM) | coverage (bigWig) |
|---|---|---|
| target field | `tpm: <column name>` | `bw: bw` |
| where the data is read from | the `csv` / `tsv` / `CPM` / `tpm` column in file_mappings | the `forward_bw` and `reverse_bw` columns |
| typical `num_before` | `512` | `0` |
| transform | `transform_targets_tpm` | `transform_targets_bw` |

Tabular:

```yaml
valid_dataset_human:
  _target_: downstream_tasks.expression_prediction.expression_dataset_final.ExpressionDatasetFixedDesc
  forward_intervals_path: "${HOME_PATH}/.../intervals/human.valid.forward.csv"
  reverse_intervals_path: "${HOME_PATH}/.../intervals/human.valid.reverse.csv"
  targets_path: "${HOME_PATH}/.../datasets/data/file_mappings/<...>_qnorm.csv"
  genome: "${HOME_PATH}/.../datasets/data/genomes/hg38/hg38.fa"
  tpm: csv
  num_before: 512
  gen_max_seq_len: 1024
  transform_targets_tpm:
    _target_: downstream_tasks.expression_prediction.expression_dataset_final.logtransform
    pseudocount: 1
```

Coverage — the same required fields, but a different target and transform:

```yaml
train_dataset_atac_human:
  _target_: downstream_tasks.expression_prediction.expression_dataset_final.ExpressionDatasetFixedDesc
  forward_intervals_path: "${HOME_PATH}/.../intervals/human.train.cre_modernbert.100k.csv"
  reverse_intervals_path: "${HOME_PATH}/.../intervals/human.train.reverse.cre_modernbert.100k.csv"
  targets_path: "${HOME_PATH}/.../datasets/data/<DATASET>/file_mappings_qnorm.csv"
  genome: "${HOME_PATH}/.../datasets/data/genomes/hg38/hg38.fa"
  bw: bw
  num_before: 0
  gen_max_seq_len: 1024
  token_len_for_fetch: 15
  transform_targets_bw:
    _target_: downstream_tasks.expression_prediction.expression_dataset_final.multiplytransform
    coefficient: 1.5
```

#### Dataset class

| `_target_` | When |
|---|---|
| `ExpressionDataset`, `ExpressionDatasetFixedDesc` | the basic ones; **at least one such block must be present in both train and valid** — see 4.4 |
| `ExpressionDatasetMode2`, `ExpressionDatasetMode2FixedDesc` | the dataset has few tracks (4.4) |
| `MethylationDataset` | methylation; `norm_bw` is forbidden for it |

`*FixedDesc` are the variants with fixed descriptions. Only `ExpressionDataset` and `ExpressionDatasetFixedDesc` take part in the `n_keys` computation; Mode2 and methylation are ignored (4.4).

#### Remaining fields

| Field | Meaning |
|---|---|
| `token_len_for_fetch` | estimate of the average token length when pulling a sequence out of the genome |
| `transform_targets_*` | `logtransform(pseudocount)` or `multiplytransform(coefficient)`; `coefficient` is used to manually balance datasets against each other |
| `norm_bw` | divide the signal by the total coverage from the metadata; `False` by default |
| `hash_prefix` | where to put the caches; `<intervals folder>/dataset_hash` by default |
| `repeat_factor` | how many times to repeat the block in train — dataset upsampling; `1` by default |

Values shared by all blocks go into `shared_dataset_params` — they are merged into every block recursively and **only for keys that are missing**, so a value written directly in a block always wins.

### 4.4. `n_keys` — the size of one training example

One example is `n_keys` gene × track pairs. Which side of the pair is fixed and which is iterated over depends on the class:

| Class | One example | Total examples |
|---|---|---|
| `ExpressionDataset`, `ExpressionDatasetFixedDesc` | one gene, `n_keys` tracks — a single sequence repeated `n_keys` times | genes × (tracks ÷ `n_keys`, rounded up) |
| `ExpressionDatasetMode2`, `ExpressionDatasetMode2FixedDesc` | one track, `n_keys` genes — `n_keys` different sequences | tracks × (genes ÷ `n_keys`, rounded up) |

Rounded up, because the last chunk may be incomplete: 45 tracks with `n_keys = 4` give 12 chunks, the last one holding a single track with the rest filled by masked padding.

The value is shared across the whole run: batches from all datasets must have the same shape, no matter which axis each of them is chunked along.

#### Where the number comes from

It is usually absent from the config — the script works it out itself. It takes the blocks whose class is `ExpressionDataset` or `ExpressionDatasetFixedDesc`, reads their `targets_path`, counts the unique `id` values and picks the **minimum**:

```
A (ExpressionDataset):          60 tracks
B (ExpressionDatasetFixedDesc): 45 tracks
C (ExpressionDataset):           3 tracks  →  n_keys = 3
D (ExpressionDatasetMode2):      2 tracks  →  does not take part
```

In the log: `[n_keys] inferred from ExpressionDataset only: min([60, 45, 3]) = 3`.

The selection goes by class name; inheritance is not taken into account:

```python
return _target_class_name(cfg) in ("ExpressionDataset", "ExpressionDatasetFixedDesc")
```

The Mode2 variants and `MethylationDataset` are not included. It is computed twice and independently — over `train_dataset*` and over `valid_dataset*` — so a suitable block is needed in both. A config made entirely of Mode2 will not start training:

```
ValueError: Cannot infer n_keys: среди train_dataset* нет ExpressionDataset.
```

> **The versions diverge.** The two-class list exists on `front`; the copy on `dna` still has the old `== "ExpressionDataset"`, where `*FixedDesc` blocks do not pass the filter and configs such as `final_4096_full.yaml` fail with this error. `run_expression_finetuning_final_hf.py` on `front` also still has the old variant. If you hit the error, check that line in your own copy.

#### What follows from this

**A small dataset drags the whole run down with it.** In the example above, block C with its three tracks sets `n_keys = 3` for everyone. No data is lost: A and B will hand over their 60 and 45 tracks in chunks of 3, there will simply be more examples. But within one example the model sees fewer relationships between tracks.

What to do with such a dataset: merge it with another, drop it from train, or switch it to Mode2 — which neither takes part in the computation nor pads examples out when there are fewer tracks than `n_keys`.

**Setting it manually** is possible via `n_keys` in `shared_dataset_params`, with two caveats: the value is accepted only if it is **smaller** than the inferred one (otherwise `keeping inferred for TRAIN` goes to the log), and it does not save you from the error above — inference runs first. The resulting number is then applied to all blocks, Mode2 included.

**`--n_keys` in `precompute_datasets.py`** is a different thing: there the value is needed to construct the dataset during warm-up. It does not affect the cache contents and is not part of the hash (section 5).

### 4.5. `optimize_metric` — what you can optimize for

Metric names are assembled from the `dataset_description` **column of file_mappings**, not from the block name in the config — hence the space in the middle of the string:

```yaml
optimize_metric: "score_predictions_Expression_dataset_v1_GRCh38_csv dataset"
```

Available: `pearson_corr_cells_*`, `pearson_corr_genes_*`, `mean_residual_<k>_*` and `score_predictions_*`.

**Limitation:** `score_predictions_*` is computed only for the two descriptions hardcoded in `ALLOWED` in `run_expression_finetuning_final.py`: `Expression_dataset_v1_GRCh38_csv dataset` and `Expression_dataset_v1_mm10_CPM dataset`. For anything else the metric never appears, and `save_best` silently fails to save the best checkpoint. With your own valid dataset, use `pearson_corr_*` or add your description to `ALLOWED`.

### 4.6. Auto-generating a multi-species config

When there are dozens of assemblies, dataset blocks are not written by hand:

```bash
cd "$EP/datasets/src"

# split by assembly and signal type
python split_file_mappings.py ../data/<DATASET_NAME>/file_mappings_qref.csv --min-rows 13

# assemble the yaml
python generate_config.py ../data/<DATASET_NAME>/file_mappings_<dataset_description>_min_13 --no-with-valid
```

`split_file_mappings.py` requires the **exact** columns `dataset_description, genome, metadata, tpm, cpm, bw`; rows with neither `tpm` nor `bw` are skipped. It creates a folder `file_mappings_<dataset_description>_min_<N>/` with files `<ds>_<genome>_<bucket>.csv`, where bucket is `tpm` / `bw` / `tpm_bw`. The docstring says "strictly greater", but the skip condition in the code is `len < min_rows`, so a group of exactly `min_rows` rows is kept.

`generate_config.py` writes `configs/config_<folder name without the file_mappings_ prefix>.yaml`, sets `tpm`/`CPM`/`bw` on its own and assigns Mode2 to CSVs with fewer rows than `--min-rows` (8 by default). It generates interval paths from the template `intervals/{genome}.{split}.{strand}.csv` — those must already exist. The result is then edited by hand: model, lr, valid, `optimize_metric`.

---

## 5. Precompute — warming up the caches

Without it the first steps go into tokenization and signal reading, and very slowly.

```bash
cd "$EP/src"

GENALM_HOME="$GENALM_HOME" python precompute_datasets.py \
  --experiment_config "$EP/configs/<config_name>.yaml" \
  --workers 120 \
  --include_valid \
  --n_keys 14
```

| Argument | Default | Meaning |
|---|---|---|
| `--experiment_config` | required | path to the yaml |
| `--workers` | ~half the CPUs | parallel processes |
| `--include_valid` | off | also warm up `valid_*` |
| `--n_keys` | 13 | forced `n_keys` for all datasets |
| `--log_file`, `-v` | — | log to file / DEBUG level |

Caches are written next to the intervals, in `$EP/intervals/`:

| File | What | The hash depends on |
|---|---|---|
| `dataset_hash.<hash>.h5` | tokens | intervals, genome, `num_before`, `token_len_for_fetch` |
| `dataset_hash.signal.<hash>.h5` | signals | the same plus `targets_path`, `gen_max_seq_len`, `norm_bw`, the set of `id`s |
| `dataset_hash.tpm.<hash>.h5` | TPM tables | same as signals |

So changing `gen_max_seq_len` or `targets_path` requires recomputing the signals but **not** the tokens; changing `num_before`, the intervals or the genome requires recomputing everything. The caches run into many gigabytes and are not cleaned up automatically — keep an eye on disk space. The cache location can be overridden with the `hash_prefix` field in a dataset block.

---

## 6. Launching

### 6.1. The script

`finetune_expression.sh` lives in `$EP`; inside, it moves to the `GENA_LM` root and launches accelerate:

```bash
#!/usr/bin/env bash
set -e
cd ../..                      # from expression_prediction to the GENA_LM root

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0,1

TBS=80          # total batch
BS=2            # batch per process
NP=2            # number of processes = number of GPUs
GAS=$(( TBS / (BS * NP) ))

config_name="<config_name>"

GENALM_HOME=$(realpath ..) accelerate launch \
  --main_process_port 29515 \
  --num_processes "$NP" \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"
```

Only `config_name`, `CUDA_VISIBLE_DEVICES`, `TBS`/`BS`/`NP` and the port are edited here. Everything else lives in the yaml.

Two requirements: `NP` must match the number of devices in `CUDA_VISIBLE_DEVICES`, and `TBS` must divide evenly by `BS * NP`, otherwise integer division will quietly change the effective batch. On a single card: `CUDA_VISIBLE_DEVICES=0`, `NP=1`.

### 6.2. Running it

```bash
cd "$EP" && nohup bash finetune_expression.sh > "logs_<config_name>.log" 2>&1 &
```
---

## 7. Checklist for a new dataset

1. Lay out the data: `datasets/data/<NAME>/{file_mappings.csv, metadata/, tpm|cpm|bw/}` (2.1).
2. Check `file_mappings.csv`: required columns, relative paths, `dataset_description` filled in (2.2).
3. Check the metadata: cell/tissue type, assay, genome (2.3).
4. Make sure the genome and the intervals for the assembly exist (2.4).
5. Run qnorm: tables — 3.1, bigWig — 3.2.
6. Add the dataset block to the config with a **normalized** `targets_path` (4.3).
7. Check that `optimize_metric` is actually computed for your `dataset_description` (4.5).
8. Estimate `n_keys`: will the new dataset lower the minimum for the others (4.4).
9. `precompute_datasets.py --include_valid` (5).

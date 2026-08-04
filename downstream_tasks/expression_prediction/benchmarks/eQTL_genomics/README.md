# TSS saturation mutagenesis

This benchmark scores every possible SNV in the configured mutation interval
(by default `[TSS-1000, TSS+1001)`, 2,001 bp) around each hg38 TSS. Every reference and alternate is
evaluated in genomic-forward and reverse-complement orientations. The output
stores the absolute reference ATAC sum, absolute mutant ATAC sum, and their
difference over `[TSS-500, TSS+501)` for both orientations.

The JSON description is rendered and tokenized with the same code path as
`ExpressionDataset`: no padding, truncation at the checkpoint YAML's
`shared_dataset_params.text_max_seq_len`, and the dataset's metadata cleanup and
sentence formatting.

SOX2 notebook reproduction code, configurations, results, and analysis are
isolated under [`reproducibility/`](reproducibility/README.md). They are not
part of the production runner described here.

## Checkpoint-local configuration requirement

The checkpoint directory must contain exactly one `*.yaml` file beside the
checkpoint `.bin`. That training YAML is authoritative for `model_kwargs`, the
DNA and description tokenizers, and `shared_dataset_params.text_max_seq_len`.
Missing or multiple checkpoint YAML files stop the run. `inference_config.yaml`
contains the default checkpoint path, mutation radius, and ATAC scoring radius;
these can still be overridden with `--checkpoint`, `--mutation-window-bp`, and
`--score-window-bp`. The
checkpoint-local YAML path, hash, resolved tokenizers, and text length are stored
in HDF5 provenance.

The default checkpoint has these equivalent names and locations:

- Local experiment name: `expression_model_v1-1`
- Model alias: `all_datasets_2`
- S3 run: `s3://genalm/expr/runs/model_010726/`

Its checkpoint-local configuration must be `final_02062026.yaml`, sourced from
that S3 run and placed beside the local `pytorch_model.bin`.

All persistent model names live under the repository-root `models/` directory:
`model_010726`, `moderngena_large`, and `decoderdp0.1`. Git creates these as
empty directories; each user must populate them locally with files or symlinks.
Likewise, users must provide `data/genomes/hg38/hg38.fa` and its
`hg38.fa.fai` index. Large model and genome files or symlinks are deliberately
not committed. Set `GENALM_HOME` to the repository root for every command; the
inference and checkpoint YAMLs resolve models through that variable.

The default condition description is the benchmark-relative
`data/descriptions/ENCFF556YQA.json`. Use `--description-json` only to override
that condition.

## Pilot

Run a two-TSS pilot first. Pick an unused GPU and an output directory:

```bash
GENALM_HOME=/path/to/GENA_LM \
CUDA_VISIBLE_DEVICES=0 \
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_saturation_mutagenesis.py \
  pilot --output-dir /path/to/pilot --device cuda:0
```

The progress line reports observed variants per second. Use it to estimate the
full run before launching production.

For the checked-in 10-TSS smoke set (six plus-strand and four minus-strand TSSs),
override the pilot's default two-record limit:

```bash
GENALM_HOME=/path/to/GENA_LM \
CUDA_VISIBLE_DEVICES=0 \
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_saturation_mutagenesis.py \
  pilot \
  --catalog downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/data/smoke_test_10_tss.tsv \
  --output-dir /path/to/smoke-output \
  --device cuda:0 \
  --limit 10
```

## Multi-GPU run and merge

```bash
GENALM_HOME=/path/to/GENA_LM \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_saturation_mutagenesis.py \
  run --output-dir /path/to/shards --devices 0,1,2,3,4,5,6

GENALM_HOME=/path/to/GENA_LM \
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_saturation_mutagenesis.py \
  merge --output-dir /path/to/shards --output /path/to/tss_ism.h5
```

Each GPU writes one deterministic shard and skips completed TSSs when restarted.
The merge command requires every shard to be complete and provenance-compatible.
Checkpoint, FASTA, batch size, CPU worker count, and prefetch depth are
command-line options; see `--help`. Model and tokenizer settings come from the
checkpoint-local training YAML.

The checkpoint's `args_params.input_seq_len` is used as the total model input
length. It includes CLS and SEP. The remaining DNA-token budget is divided
evenly around the TSS; for every current 1,024-position checkpoint this is one
CLS, 511 upstream DNA tokens, 511 downstream DNA tokens, and one SEP. Near a
chromosome end, less source context may be available. The total configured
length and both derived side lengths are recorded in provenance.

The final HDF5 stores flat `scores[N,3,2]`, `variant_atac_sum[N,3,2]`,
`ref_base[N]`, and `position_offset[N]` arrays plus `/tss` offsets and metadata.
The last axis is `[forward, reverse_complement]`. Alternative bases are the
lexicographically ordered members of `ACGT` excluding the reference.
Use `iter_variant_scores()` from `saturation_mutagenesis.py` to obtain explicit
1-based genomic variant records.

## Explicit variant catalog scoring

`score_variant_catalog.py` is a separate workflow for `data/variant_catalog.tsv`.
It scores exactly one supplied REF/ALT pair per row, including SNVs, insertions,
and deletions. Both the model input and the ATAC aggregation window are centered
on the row's 1-based variant position (the first REF base). The stored float32
`score` is:

```text
ALT ATAC sum over [variant-500, variant+501)
  - REF ATAC sum over [variant-500, variant+501)
```

Run a ten-variant pilot, then a multi-GPU production run and merge:

```bash
GENALM_HOME=/path/to/GENA_LM CUDA_VISIBLE_DEVICES=0 \
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/score_variant_catalog.py \
  pilot --output-dir /path/to/variant-pilot --device cuda:0 --limit 10

GENALM_HOME=/path/to/GENA_LM CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/score_variant_catalog.py \
  run --output-dir /path/to/variant-shards --devices 0,1,2,3

GENALM_HOME=/path/to/GENA_LM \
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/score_variant_catalog.py \
  merge --output-dir /path/to/variant-shards --output /path/to/variant-scores.h5
```

### Seven-GPU tmux example

The following one-liner starts a detached `tmux` session using GPUs 0–6 and
writes the combined worker output to `outputs/variant_catalog/run.log`:

```bash
tmux new-session -d -s variant_catalog "zsh -lc 'cd /home/jovyan/minja/DNALM/GENA_LM && mkdir -p outputs/variant_catalog && GENALM_HOME=\$PWD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 /home/jovyan/miniconda3/envs/api/bin/python downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/score_variant_catalog.py run --output-dir outputs/variant_catalog/shards --devices 0,1,2,3,4,5,6 2>&1 | tee outputs/variant_catalog/run.log'"
```

Each worker prints its completed and pending variant counts and observed
variants per second. Attach to the session to monitor it interactively:

```bash
tmux attach -t variant_catalog
```

Detach without stopping the run with `Ctrl-b`, then `d`. Alternatively, follow
the log without attaching:

```bash
tail -f /home/jovyan/minja/DNALM/GENA_LM/outputs/variant_catalog/run.log
```

After all workers finish, merge the seven shards:

```bash
cd /home/jovyan/minja/DNALM/GENA_LM && GENALM_HOME=$PWD /home/jovyan/miniconda3/envs/api/bin/python downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/score_variant_catalog.py merge --output-dir outputs/variant_catalog/shards --output outputs/variant_catalog/variant-scores.h5
```

### Expected merged output

The merge produces one HDF5 file with one row per input catalog row, restored to
the original catalog order. For the complete checked-in catalog, every dataset
below therefore has 75,920 rows. The row-aligned datasets are:

| Dataset | Type | Meaning |
| --- | --- | --- |
| `catalog_index` | int64 | Zero-based row number in the input TSV. |
| `variant_id` | string | Catalog identifier, `chrom-position-ref-alt`. |
| `chromosome` | string | hg38 chromosome. |
| `position_1based` | int64 | Catalog position; the first REF base and scoring center. |
| `reference`, `alternate` | string | Exact alleles used for the two predictions. |
| `variant_type` | string | `SNV`, `insertion`, or `deletion`. |
| `score` | float32 | Alternative ATAC sum minus reference ATAC sum. |
| `present_in_train`, `present_in_validation` | bool | Original catalog membership flags. |
| `train_signal_count`, `validation_signal_count`, `total_signal_count` | int32 | Original catalog counts. |
| `status` | uint8 | `1` for a completed row. Merge rejects incomplete shards, so every merged row is `1`. |
| `error` | string | Empty for successfully merged rows. |

The merged file's root attributes provide provenance: schema version; hashes of
the catalog, checkpoint, checkpoint YAML, FASTA index, and description JSON;
total DNA input length and upstream/downstream token allocation; fetched context
length; local variant center; scoring-window definition; score sign; and shard
count. `shard_index` is intentionally removed during merge.

A positive `score` means the ALT sequence increased the predicted ATAC sum in
the configured window; a negative value means it decreased it. The file does
not store separate absolute REF and ALT sums—only their difference.

Example inspection:

```bash
cd /home/jovyan/minja/DNALM/GENA_LM && /home/jovyan/miniconda3/envs/api/bin/python -c 'import h5py; f=h5py.File("outputs/variant_catalog/variant-scores.h5"); print(dict(f.attrs)); print({k: f[k].shape for k in f}); print(f["variant_id"][0], f["score"][0])'
```

The genomic flanks come from hg38. The sequence at the variant interval is set
to the catalog REF for the reference prediction and to catalog ALT for the
alternative prediction. This preserves the requested `ALT - REF` direction even
when a catalog REF label differs from the hg38 allele. The runner is resumable
at completed batches and retains the original catalog metadata in each shard
and in the merged file. Defaults are in `variant_inference_config.yaml`;
checkpoint model and tokenizer settings continue to come from the single YAML
beside the checkpoint.

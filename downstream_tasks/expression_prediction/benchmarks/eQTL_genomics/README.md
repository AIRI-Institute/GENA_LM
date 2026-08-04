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

## SOX2 notebook validation

`data/sox2_tss.tsv` contains the notebook-equivalent plus-strand SOX2 TSS at
`chr3:181711925` (hg38, 1-based; notebook coordinate 181711924 is 0-based). Keep its profiling artifacts separate from
the general smoke test:

The dedicated `sox2_inference_config.yaml` selects the `dev_loss` model
(locally backed by `expression_model_v1-2`), the notebook's H1
description `ENCFF081FQX`, and a reduced ±600-bp mutation interval. Scoring
remains fixed at TSS ±500 bp; it is not variant-centered.

```bash
cd "${GENALM_HOME}" && CUDA_VISIBLE_DEVICES=0 /home/jovyan/miniconda3/envs/api/bin/python downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_saturation_mutagenesis.py pilot --inference-config downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_inference_config.yaml --catalog downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/data/sox2_tss.tsv --output-dir downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_model_270526_output --device cuda:0 --limit 1 2>&1 | tee downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_model_270526_profile.log
```

The one-worker shard is already a complete HDF5 result. To give it a dedicated
final filename and draw the forward-orientation 600-bp upstream promoter used
by the API notebook:

```bash
cd "${GENALM_HOME}" && /home/jovyan/miniconda3/envs/api/bin/python downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_saturation_mutagenesis.py merge --output-dir downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_output --output downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_saturation_mutagenesis.h5 && /home/jovyan/miniconda3/envs/api/bin/python downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/plot_ref_alt.py downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_saturation_mutagenesis.h5 --output downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_forward_promoter_ref_alt.png --orientation forward --relative-start -600 --relative-end -1 --y-min -500 --y-max 500 --title "SOX2 promoter — forward"
```

The helper reads the stored `scores` (alternative minus reference ATAC sum).
Alternative alleles are colors and reference alleles are marker shapes, matching
the notebook convention. Use `--orientation reverse_complement` or omit the
relative bounds to inspect the other orientation or the entire ±1000-bp window.

For notebook-style scoring, use `run_variant_centered_mutagenesis.py`. It keeps
DNA tokenization centered at the TSS but scores each allele over a 501-bp ATAC
window centered on that variant. Its HDF5 additionally stores
`reference_atac_sum_by_position[N,2]`, because the reference score changes with
the genomic position:

```bash
cd "${GENALM_HOME}" && CUDA_VISIBLE_DEVICES=0 /home/jovyan/miniconda3/envs/api/bin/python downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_variant_centered_mutagenesis.py pilot --inference-config downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_inference_config.yaml --catalog downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/data/sox2_tss.tsv --output-dir downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/sox2_model_270526_h1_variant_centered_output --device cuda:0 --limit 1
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

The checkpoint's `args_params.input_seq_len` is the model capacity. The actual
experiment input is controlled by `--dna-input-seq-len` (default 1,022) and must
not exceed that capacity. Two positions are reserved for CLS and SEP, so with
`--num-before 510` the remaining 510 positions are downstream DNA tokens. The
actual input length and both derived side lengths are recorded in provenance.

The final HDF5 stores flat `scores[N,3,2]`, `variant_atac_sum[N,3,2]`,
`ref_base[N]`, and `position_offset[N]` arrays plus `/tss` offsets and metadata.
The last axis is `[forward, reverse_complement]`. Alternative bases are the
lexicographically ordered members of `ACGT` excluding the reference.
Use `iter_variant_scores()` from `saturation_mutagenesis.py` to obtain explicit
1-based genomic variant records.

# TSS saturation mutagenesis

This benchmark scores every possible SNV in the fixed 2,001-bp interval
`[TSS-1000, TSS+1001)` around each hg38 TSS. Every reference and alternate is
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
contains only the default checkpoint path, which can still be overridden with
`--checkpoint`. The checkpoint-local YAML path, hash, resolved tokenizers, and
text length are stored in HDF5 provenance.

## Pilot

Run a two-TSS pilot first. Pick an unused GPU and an output directory:

```bash
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_saturation_mutagenesis.py \
  pilot --description-json /path/to/description.json \
  --output-dir /path/to/pilot --device cuda:0
```

The progress line reports observed variants per second. Use it to estimate the
full run before launching production.

## Multi-GPU run and merge

```bash
/home/jovyan/miniconda3/envs/api/bin/python \
  downstream_tasks/expression_prediction/benchmarks/eQTL_genomics/run_saturation_mutagenesis.py \
  run --description-json /path/to/description.json \
  --output-dir /path/to/shards --devices 0,1,2,3,4,5,6

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

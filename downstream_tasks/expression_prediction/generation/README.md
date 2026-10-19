# GENA sequence generation

This project samples promoter-length sequences from an annotated genome, pastes
one sampled sequence into a plasmid by replacing a configured GenBank feature,
and evolves that region with `gena_expression.SequenceOptimizer`.

The sampling and plasmid steps are independent:

1. `PromoterSampler` reads only the configured genome FASTA and transcript
   annotation.
2. The project samples `optimization.population_size` promoters.
3. Sample zero replaces the named plasmid feature and becomes the optimizer's
   reference region.
4. Samples one through `population_size - 1` become `initial_population`.

Genomic features are not copied into the plasmid because they would become
stale after evolution. Their chromosome, coordinate, strand, transcript, and
gene provenance is saved separately with the results.

## Layout

```text
gena_sequence_generation/
├── config.yaml
├── run_generation.py
├── plasmids/
├── on_target/
├── off_target/
├── results/
├── utils/
└── tests/
```

`run_generation.py` contains only CLI orchestration. Configuration validation,
input construction, component factories, experiment execution, and result
serialization live in small modules under `utils/`.

## Installation

Install the local API checkout first, then this project:

```bash
python -m pip install -e "/path/to/gena_expression_refactored[all]"
python -m pip install -e .
```

The API checkout used while building this project was:

```text
C:\Users\Lenovo\Documents\Codex\2026-07-21\i-have-an-expression-model-api-3\outputs\gena_expression_refactored
```

## Configuration

Copy or edit `config.yaml`. Every generation-affecting value is explicit.
Filesystem paths are resolved relative to the configuration file, so this is
portable:

```yaml
plasmid:
  genbank_path: plasmids/my_plasmid.gb
  fasta_path: null
```

Tokenizer values support three forms:

- `null`: allow `SequenceModel.load()` to use its model-config fallback;
- `{path: relative/or/absolute/path}`: a local resource;
- `{identifier: Qwen/Qwen3-Embedding-0.6B}` or a plain string: a model-hub
  identifier.

The two condition directories are intentionally separate. Each role has an
explicit list of lookup keys. `DescriptionLookup` scans the corresponding flat
directory for `*.json` files and selects matches deterministically.

The configured plasmid replacement feature is matched by exact feature name.
Zero or multiple matches fail before model loading. When both plasmid FASTA and
GenBank files are configured, their DNA sequences must agree. The replacement
feature must not overlap the reporter TSS.

`plasmid.window_size: full` preserves the current runner's full-plasmid context.
An integer requests a fixed reporter-centered window.

Supported component types are:

- scorer: `expression`, `track_feature`, `track_window`, or
  `linear_combination`;
- fitness: `mean_on_minus_mean_off` or `min_on_max_off`;
- offspring: `sequential` or `parallel`.

For a linear combination, each component contains a nested scorer and one
coefficient:

```yaml
scorer:
  type: linear_combination
  params:
    name: combined_activity
    components:
      expression:
        coefficient: 1.0
        scorer:
          type: expression
          params: {}
      promoter_track:
        coefficient: 0.002
        scorer:
          type: track_feature
          params:
            features: promoter
            aggregate: sum
```

## Running

```bash
python run_generation.py --config config.yaml
```

The parallel offspring generator and process-based preprocessing are safe
because the executable work remains behind the script's `__main__` guard.

## Outputs

The original runner's outputs are preserved:

- `evaluations.{parquet,csv}`
- `final_population.{parquet,csv}`
- `history.{parquet,csv}`
- `generation_min_on_max_off.{parquet,csv}`
- `summary.json`
- `run_config.json`
- `output_manifest.json`
- `evaluated_region_sequences.fasta`
- `final_population_region_sequences.fasta`
- `final_population_full_sequences.fasta`
- `best_full_sequence.fasta`

Two additional files make genome sampling auditable:

- `initial_population.fasta`
- `initial_population_metadata.json`

The manifest records the actual configured region length; it does not contain a
hard-coded 600 bp description.

By default, the runner refuses to write into a non-empty result directory.
Set `output.allow_existing: true` only when overwriting the known files is
intentional.

## Tests

From the project directory, with the API installed:

```bash
python -B -m unittest discover -s tests -v
```

The test suite uses synthetic genomes, conditions, plasmids, optimizer tables,
and models. It does not require the real checkpoint or a GPU.

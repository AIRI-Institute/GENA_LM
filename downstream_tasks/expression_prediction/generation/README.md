# GENA sequence generation

This project samples promoter-length sequences from an annotated genome, pastes
one sampled sequence into a plasmid by replacing a configured GenBank feature,
and evolves that region with `gena_expression.SequenceOptimizer`.

The sampling and plasmid steps are independent:

1. `PromoterSampler` reads only the configured genome FASTA and transcript
   annotation.
2. The project samples `optimization.initial_population_size` promoters.
3. Sample zero replaces the named plasmid feature and becomes the optimizer's
   reference region.
4. The remaining samples become `initial_population`.
5. Generation zero evaluates all sampled candidates and reduces them to
   `optimization.population_size` using the configured selection strategy.

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

Initial and retained population sizes are configured separately:

```yaml
optimization:
  initial_population_size: 30000
  population_size: 800
```

`initial_population_size` must be at least `population_size`, and
`sampler.max_attempts` must be at least `initial_population_size`. Generation
zero evaluates all unique sampled insertions before selection. Consequently, a
large initial population can dominate the total runtime even when later
generations are small.

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

Population snapshots are controlled by an explicit list:

```yaml
snapshots:
  generations: [0, 200, 400, 600, 800, 1000]
```

Each requested generation is saved after scoring and selection under
`results/<run>/snapshots/generation_XXXXXX/`. Use an empty list to disable
snapshots. Each directory contains:

- `population.{parquet,csv}` with insertion sequence, rank, lineage, fitness,
  and target scores;
- `insertion_sequences.fasta`;
- `metrics.json` with fitness summaries, the best candidate's target scores,
  and population complexity metrics.

### Resuming from the latest snapshot

Resume is a warm restart from the latest complete selected population:

```yaml
resume:
  from_latest_snapshot: true
  source_directory: null
  genome_sequences_to_add: 0
```

`source_directory: null` uses `output.directory`. A different experiment
directory may be supplied when the resumed run should write into a new output
directory. A snapshot is complete when both `population.parquet` and
`metrics.json` exist; newer partial directories are ignored.

The loader validates insertion length, alphabet, uniqueness, candidate IDs,
population size, generation number, and the absence of plasmid sequences. If
the source contains `run_config.json`, model, plasmid, condition, scorer,
fitness, offspring, and population settings must be compatible. Older
snapshots without that file remain usable and are marked as unverified in
`resume.json`.

Restored insertions are placed back into the configured plasmid and rescored.
For a snapshot at generation 200, offspring start at generation 201.
`optimization.n_generations` is the absolute final generation, so a value of
1000 runs through generation 1000 rather than adding another 1000 generations.
The source generation snapshot is not overwritten.

Resume does not expand the selected population back to
`optimization.initial_population_size`. Set `genome_sequences_to_add` to a
positive number only when intentionally injecting additional genome-sampled
promoters. Those additions are scored with the restored population before it
is selected back down to `optimization.population_size`.

With `top_k` selection and deterministic model execution, generation-indexed
offspring seeds allow a resumed run to match an uninterrupted run from the
same population. `tournament` and `rank_weighted` restart the optimizer RNG
stream because optimizer state is deliberately not saved.

## Outputs

All sequence-bearing project outputs contain insertion sequences only. Full
materialized plasmid sequences are used for scoring but are not written.

- `evaluations.{parquet,csv}`
- `final_population.{parquet,csv}`
- `history.{parquet,csv}`
- `generation_min_on_max_off.{parquet,csv}`
- `summary.json`
- `run_config.json`
- `output_manifest.json`
- `evaluated_region_sequences.fasta`
- `final_population_region_sequences.fasta`
- `best_insertion_sequence.fasta`

A resumed run writes `resume_config.json` and `resume.json` instead of
overwriting the source `run_config.json`. When genomic additions are enabled,
it also writes `resume_genome_additions.fasta` and
`resume_genome_additions_metadata.json`. Root-level evaluations and history
cover the restored population and later generations; earlier populations
remain available in their original snapshot directories.

In `history.{parquet,csv}`, `best_candidate_id`, `best_fitness`,
`best_min_on_target`, and `best_max_off_target` all describe the same
best-fitness sequence for that generation. The compact
`generation_min_on_max_off` tables include that candidate ID and fitness as
well as its target metrics.

Two additional files make genome sampling auditable:

- `initial_population.fasta`
- `initial_population_metadata.json`

The manifest records the actual configured region length; it does not contain a
hard-coded 600 bp description. It also records that materialized plasmids are
absent from sequence exports.

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

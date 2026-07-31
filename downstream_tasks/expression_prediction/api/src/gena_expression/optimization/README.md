# Sequence optimization

The `gena_expression.optimization` subpackage evolves fixed-length replacements
for one annotated sequence region. It scores every candidate under configured
on-target/off-target conditions through `SequenceModel.predict_multiple_sequences()`,
converts those absolute scores to fitness, selects a population, and records the
run in Polars tables.

`SequenceOptimizer` is the only public class in this folder.

## Module map

| Module | Contents |
| --- | --- |
| `core.py` | `SequenceOptimizer` and its internal Polars dependency loader |
| `__init__.py` | Exports `SequenceOptimizer` |

Read the package [README](../README.md) for `AnnotatedSequence`/`Feature`.
Read [`inference/README.md`](../inference/README.md) for batching arguments and
[`scoring/README.md`](../scoring/README.md) for absolute scorer contracts.

## Dependencies and core constraints

- Construction immediately imports Polars and initializes public DataFrames.
- `sequence` must be a non-empty `AnnotatedSequence`.
- The evolvable region has a fixed length. The original sequence, every initial
  candidate, and every generated candidate use exactly that region length.
- Candidate strings accept uppercase/lowercase `A`, `C`, `G`, `T`, and `N` and
  are normalized to uppercase.
- The scorer must implement `score_prediction(prediction)` and return one finite
  real scalar for every sequence/condition row.
- The model passed to `run_evolution()` must implement
  `predict_multiple_sequences(...)`.
- Because replacement length is fixed, materialized candidates preserve the
  source sequence's features and coordinate-map object.

## `SequenceOptimizer`

### Constructor

```python
SequenceOptimizer(
    sequence,
    *,
    region,
    tokenization_center,
    scorer,
    on_targets,
    off_targets=(),
    offspring_generator,
    fitness_function=None,
    population_size=64,
    offspring_size=None,
    n_generations=20,
    initial_population=None,
    selection_strategy="top_k",
    elite_count=1,
    tournament_size=3,
    offspring_generator_mode="sequences",
    random_seed=0,
)
```

### Sequence, region, and scoring arguments

| Argument | Purpose and validation |
| --- | --- |
| `sequence: AnnotatedSequence` | Full reference input. It is never mutated. |
| `region: tuple[int, int] \| str \| Feature` | Non-empty evolvable interval inside `sequence`. A tuple is `[start, end)`. A string uses an exact unique feature name/type and may start with `feature:`. |
| `tokenization_center: int \| str \| Feature` | Passed unchanged as `center=` to `model.predict_multiple_sequences()` for every materialized candidate. |
| `scorer` | Absolute scorer with callable `score_prediction()`. `ExpressionScorer` and the coordinate-aware track scorers are compatible when they return a scalar. `TrackAllFeaturesScorer` returns a mapping and is therefore not compatible. |
| `on_targets` | Mapping `{label: condition}` or non-string condition sequence. |
| `off_targets` | Same accepted forms; may be empty. |
| `fitness_function` | Callable `(on_scores, off_scores) -> finite real`. `None` uses mean on-target minus mean off-target. |

At least one on-target or off-target is required.

Target normalization:

- Mapping keys become labels.
- Sequence items use `condition.name` when present/non-empty, otherwise
  `on_target_<index>` or `off_target_<index>`.
- Labels must be non-empty and unique within each role.
- Public score columns are `on_target__<label>` and
  `off_target__<label>`.

The default fitness is:

```text
mean(on-target scores, or 0 if none)
-
mean(off-target scores, or 0 if none)
```

### Population and evolution arguments

| Argument | Purpose and validation |
| --- | --- |
| `offspring_generator` | Required callable. Its call signature depends on `offspring_generator_mode`; return/yield values are described below. |
| `population_size` | Positive maximum selected population size. |
| `offspring_size` | Positive maximum generator items consumed per generation. `None` uses `population_size`. Duplicate/existing sequences may make the number of newly evaluated candidates smaller. |
| `n_generations` | Non-negative integer. Generation zero scores seeds; values above zero add that many reproduction rounds. |
| `initial_population` | Optional iterable of region-only DNA strings. The original region is always inserted before these. |
| `selection_strategy` | `"top_k"`, `"tournament"`, or `"rank_weighted"`. |
| `elite_count` | Integer from `0` through `population_size`; top candidates protected before stochastic selection. |
| `tournament_size` | Positive number of available candidates sampled per tournament. |
| `offspring_generator_mode` | `"sequences"` or `"records"`. |
| `random_seed` | Seed for selection and the advanced generator context's `rng`. `reset()` recreates the same seeded RNG. |

### Offspring generator contracts

#### `"sequences"` mode

The optimizer calls:

```python
generated = offspring_generator(parent_region_sequences)
```

`parent_region_sequences` is a list in current population order. Each generated
item may be:

- a DNA string; or
- a mapping with required `"region_sequence"` and optional `"parent_ids"`.

When the population has exactly one parent and a generated item is a string,
that parent ID is recorded automatically. Otherwise string items have no
explicit parent IDs.

#### `"records"` mode

The optimizer calls:

```python
generated = offspring_generator(parent_records, context)
```

Each parent record contains:

| Key | Meaning |
| --- | --- |
| `candidate_id` | SHA-256 of the region sequence. |
| `region_sequence` | Region-only DNA. |
| `fitness` | Current scalar fitness. |
| `on_scores`, `off_scores` | Label-to-score dictionaries. |
| `generation_born` | First generation in which the sequence appeared. |
| `parent_ids` | Recorded lineage IDs. |

The context contains:

| Key | Meaning |
| --- | --- |
| `generation`, `n_generations` | Current and configured generation counts. |
| `n_offspring`, `population_size` | Requested generator output limit and selection size. |
| `region` | Mapping with name, start, end, and length. |
| `rng` | Optimizer's seeded `random.Random` instance. |

Generated items follow the same string/mapping rules as sequence mode.

Only the first `offspring_size` yielded items are consumed. Candidate IDs
deduplicate identical region strings within and across generations. Previously
evaluated candidates may re-enter a selection pool without being rescored.

### Selection strategies

Fitness is maximized; ties are resolved deterministically by candidate ID.

| Strategy | Behavior after protected elites |
| --- | --- |
| `"top_k"` | Take the remaining highest-fitness candidates. |
| `"tournament"` | Repeatedly sample up to `tournament_size` candidates and select the best competitor. |
| `"rank_weighted"` | Sort remaining candidates and sample without replacement using descending integer weights by rank. |

If fewer unique candidates exist than `population_size`, the selected
population is smaller.

## Running the optimizer

### `run_evolution(...) -> SequenceOptimizer`

```python
run_evolution(
    model,
    *,
    evaluation_chunk_size=128,
    grouping="auto",
    preprocessing_workers=0,
    preprocessing_backend="process",
    max_records_per_forward=None,
    prefetch_batches=1,
    show_progress=True,
)
```

| Argument | Purpose |
| --- | --- |
| `model` | Object exposing `predict_multiple_sequences()`, normally `SequenceModel`. |
| `evaluation_chunk_size` | Positive number of candidate IDs handled in one outer evaluation chunk. |
| `grouping` | `"serial"`, `"no_grouping"`, `"condition"`, `"sequence"`, or `"auto"`, forwarded to inference. `"auto"` is the default because each candidate repeats across target conditions while conditions repeat across candidates. |
| `preprocessing_workers` | Forwarded CPU worker count. |
| `preprocessing_backend` | `"process"` or `"thread"`, forwarded. |
| `max_records_per_forward` | Optional inference forward-row limit. |
| `prefetch_batches` | Inference look-ahead count. |
| `show_progress` | Enable outer chunk progress when `tqdm` is available. Inner model progress is disabled. |

Scoring flow:

1. extract the original region and combine it with `initial_population`;
2. validate/deduplicate seeds;
3. materialize every pending full sequence;
4. create one sequence/condition row for every configured target;
5. call `model.predict_multiple_sequences()`;
6. call `scorer.score_prediction()` on each returned prediction;
7. compute/validate fitness;
8. select generation zero;
9. generate, evaluate, and select each reproduction generation;
10. populate public tables.

The same materialized `AnnotatedSequence` object is reused across a candidate's
conditions so the inference layer can deduplicate tokenization.

The optimizer inspects `scorer.requires_tokens`, defaulting to `True` when the
attribute is absent:

- false: `return_tokens=False`, `retention="scalars"`;
- true: `return_tokens=True`, `retention="full"`.

An optimizer instance may run only once until `reset()` is called.

## Public tables

All three attributes are Polars DataFrames. They are empty before a completed
run.

### `evaluations`

One row per unique scored candidate:

| Column | Meaning |
| --- | --- |
| `candidate_id` | SHA-256 of region DNA. |
| `region_sequence` | Region-only candidate. |
| `generation_born` | First generation seen. |
| `source` | `"reference"`, `"initial"`, or `"offspring"`. |
| `parent_ids` | Recorded lineage list. |
| `fitness` | Validated scalar. |
| target columns | One absolute score per normalized target label. |

Sorted by `generation_born`, then `candidate_id`.

### `final_population`

Selected candidates ordered by descending fitness/candidate-ID tie break. It
adds:

| Column | Meaning |
| --- | --- |
| `sequence` | Full materialized DNA. |
| `rank` | One-based final rank. |

It also includes region DNA, lineage, fitness, and all target score columns.

### `history`

One row for generation zero and each reproduction generation:

| Column | Meaning |
| --- | --- |
| `generation` | Generation number. |
| `proposed_candidates` | Unique proposed IDs considered in that generation. |
| `evaluated_candidates` | Previously unseen IDs actually scored. |
| `population_size` | Selected population size. |
| `best_fitness`, `mean_fitness` | Population fitness summary. |

## Other public methods and properties

### `materialize(candidate_id) -> AnnotatedSequence`

Look up an evaluated candidate, replace the region in the original full
sequence, preserve existing features and coordinate map, assign a name ending
in the first 12 ID characters, and add optimization metadata.

Unknown IDs raise `KeyError`.

### `best_sequence`

Property returning the materialized highest-fitness candidate in the current
final population. It raises before a population is available.

### `top(n=20, *, table="final_population")`

`n` must be positive. Valid table values are `"final_population"` and
`"evaluations"`. Return the selected frame sorted by descending fitness and
ascending candidate ID, limited to `n`. An empty source frame is returned
unchanged.

### `summary()`

Returns:

- source sequence name;
- resolved region name/start/end;
- selection strategy and configured reproduction generations;
- number of internal candidate records;
- final population size;
- current best fitness, or `None`.

The `evaluated_candidates` key counts all internal records. After a successful
run these records are scored; before/during a failed run the count may include
unscored candidates.

### `write_parquet(path, *, table="final_population")`

Valid table values are `"final_population"`, `"evaluations"`, and `"history"`.
The method creates parent directories, writes the selected frame, and returns
the output `Path`.

### `reset() -> SequenceOptimizer`

Clear internal records/population/history, restore the configured RNG seed,
mark the instance runnable again, replace all public tables with empty Polars
frames, and return `self`.

## Example

```python
import random

from gena_expression import ExpressionScorer, SequenceOptimizer


def mutate(parents):
    rng = random.Random(11)
    for parent in parents:
        bases = list(parent)
        position = rng.randrange(len(bases))
        bases[position] = rng.choice("ACGT")
        yield "".join(bases)


optimizer = SequenceOptimizer(
    sequence,
    region="enhancer",
    tokenization_center="tss",
    scorer=ExpressionScorer(),
    on_targets={"target": target_condition},
    off_targets={"off_target": off_target_condition},
    offspring_generator=mutate,
    population_size=32,
    n_generations=10,
    random_seed=11,
)

optimizer.run_evolution(
    model,
    grouping="auto",
    evaluation_chunk_size=64,
    show_progress=True,
)

best = optimizer.best_sequence
top_rows = optimizer.top(10)
optimizer.write_parquet("results/final_population.parquet")
```

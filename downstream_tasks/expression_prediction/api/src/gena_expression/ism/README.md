# In-silico mutagenesis

The `gena_expression.ism` subpackage generates compact single-base mutation
libraries over an annotated sequence region, materializes allele pairs lazily,
scores them through `VariantInterpreter`, and stores scalar results in a Polars
DataFrame.

`ISM` is the only public class in this folder. Plotting functions are module
helpers called by its methods.

## Module map

| Module | Contents |
| --- | --- |
| `core.py` | `ISM`, mutation generation, pair construction, scoring, queries, and Parquet export |
| `motifs.py` | MEME parsing, reference motif matching, and genomic coordinate projection |
| `attribution.py` | SNV effect matrices, peak detection, and motif profile ranking |
| `plotting.py` | Lollipop, substitution-matrix, motif-interval, and attribution-logo plotting |
| `__init__.py` | Exports `ISM` |

Coordinates follow the package-wide 0-based, half-open convention. Read the
package [README](../README.md) for `AnnotatedSequence`, `Feature`, `Variant`,
`SequencePair`, and coordinate maps. Read the
[scoring README](../scoring/README.md) for interpreter/scorer options.

## Dependencies

Constructing `ISM` immediately imports Polars and creates its public table. If
Polars is unavailable, construction raises an `ImportError` directing the user
to the project's `tables` extra.

Motif annotation uses NumPy and Polars. Plot methods also use Matplotlib.

## `ISM`

### Constructor

```python
ISM(
    sequence,
    *,
    center,
    region,
    condition,
    n_substitutions=None,
    n_deletions=None,
    random_seed=0,
    center_overlap="nearest",
)
```

| Argument | Purpose and validation |
| --- | --- |
| `sequence: AnnotatedSequence` | Non-empty reference sequence. Plain strings are rejected. It must not already have a feature whose name or type is `"variant"` and must not contain the reserved feature name `"__ism_center__"`. |
| `center: int \| str \| Feature` | Tokenization anchor used for every REF/ALT pair. An integer is a local base coordinate. A string selects one exact feature name/type (optional `feature:` prefix). A `Feature` uses its midpoint. The resolved position must lie inside the sequence. |
| `region: tuple[int, int] \| str \| Feature` | Non-empty mutation interval inside the sequence. A string is a unique exact feature name/type; a tuple is interpreted as `[start, end)`. |
| `condition` | Condition-compatible value forwarded unchanged as the singular `condition=` argument to `VariantInterpreter.score_sequence_pairs()`. |
| `n_substitutions: int \| None` | `None` creates all three alternative A/C/G/T bases at every region position whose reference is A/C/G/T. A non-negative integer samples that many unique SNVs. It may not exceed the available count. |
| `n_deletions: int \| None` | `None` or `0` creates no deletions. A positive integer samples that many distinct single-base deletions and may not exceed region length. |
| `random_seed: int \| None` | Seed for deterministic substitution sampling. Deletion sampling uses `seed + 1` when the seed is not `None`. `None` gives system-seeded randomness. |
| `center_overlap` | `"nearest"` moves an ALT tokenization center to the nearest surviving base when a deletion removes the center. `"error"` raises when that pair is materialized/scored. |

Count values reject booleans and negative numbers.

### Substitution sampling behavior

With a finite `n_substitutions`, the implementation:

1. shuffles the three non-reference alternatives independently at each
   eligible position;
2. shuffles positions for each of three rounds;
3. gives positions at most one alternative in the first round before any
   position receives a second, and similarly before a third;
4. stops at the requested count.

Positions containing symbols outside `ACGT` are skipped for substitutions but
remain eligible for a requested single-base deletion.

### Public attributes

| Attribute | Meaning |
| --- | --- |
| `sequence` | Original reference `AnnotatedSequence`. |
| `region` | Resolved `Feature` describing the ISM interval. Coordinate tuples become a feature named/type `"ism_region"`. |
| `center` | Original center selector. |
| `center_position` | Resolved integer sequence coordinate. |
| `condition` | Condition value supplied at construction. |
| `random_seed`, `center_overlap`, `n_substitutions`, `n_deletions` | Normalized constructor controls. |
| `variants` | Polars DataFrame containing structural mutation columns and any score/error columns. |
| `tf_hits` | Most recent motif-hit table, or `None` until `tf_annotate()` succeeds. |
| `scorer_names` | Property returning non-structural, non-`__error` column names. |

## Mutation-table schema

Initial columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `variant_id` | string | Stable local ID: `sub_<position>_<ref>_<alt>` or `del_<position>_<ref>`. |
| `mutation_type` | string | `"substitution"` or `"deletion"`. |
| `position` | integer | 0-based sequence-local changed base. |
| `end` | integer | `position + 1`. |
| `position_in_region` | integer | `position - region.start`. |
| `ref`, `alt` | string | Single-base REF and substituted base, or empty ALT for deletion. |
| `length` | integer | Always `1` for this single-base library. |
| `source` | string/null | `CoordinateMap` source label at the changed base. |
| `source_position` | integer/null | Mapped source coordinate, if known. |
| `source_chrom` | string/null | Mapped source chromosome/contig. |
| `source_strand` | string/null | Mapped source strand. |

Scoring adds one floating-point column per scorer. With
`on_error="record"`, it also adds `<scorer>__error` string columns.

Rows are sorted by sequence position, mutation type, and ALT.

## Variant and pair materialization

### `variant(variant_id) -> Variant`

Look up exactly one table row and construct a sequence-only `Variant`:

- `chrom=None`;
- `pos=position`;
- table REF/ALT and ID;
- mutation type, position within the region, region name, and source-coordinate
  fields in metadata.

Unknown IDs raise `KeyError`.

### `sequence_pair(variant_id) -> SequencePair`

Materializes one pair without storing it in the table:

1. add a REF `"variant"` feature;
2. call `Variant.apply_to()` for ALT;
3. replace the generated ALT marker with a valid one-base boundary marker;
4. add a private one-base `"__ism_center__"` feature independently to REF and
   ALT;
5. shift the ALT center left for a deletion before the center;
6. handle a deletion exactly at the center according to `center_overlap`;
7. return a validated pair with pair metadata.

The private center feature is why scoring can use an annotation rather than a
stale integer after deletions.

## Scoring

### `score(...) -> ISM`

```python
score(
    interpreter,
    scorer,
    *,
    column_name=None,
    chunk_size=128,
    grouping="no_grouping",
    pair_execution="separate",
    preprocessing_workers=0,
    preprocessing_backend="process",
    max_records_per_forward=None,
    max_pairs_per_forward=None,
    prefetch_batches=1,
    show_progress=True,
    overwrite=False,
    on_error="raise",
)
```

| Argument | Purpose |
| --- | --- |
| `interpreter` | Object exposing `score_sequence_pairs(...)`; normally `VariantInterpreter`. |
| `scorer` | One pair scorer or `ScorerSet`. Every resulting score must be one real scalar; list and mapping scores are rejected. |
| `column_name` | Override the single scorer's table-column name. It cannot be used with `ScorerSet`. |
| `chunk_size` | Positive number of mutation pairs materialized/submitted at once. This bounds workflow-level pair/result retention. |
| `grouping`, `pair_execution` | Forwarded to `VariantInterpreter`. |
| `preprocessing_workers`, `preprocessing_backend`, `prefetch_batches` | Forwarded CPU preparation controls. |
| `max_records_per_forward` | Forwarded limit for separate pair execution. |
| `max_pairs_per_forward` | Forwarded limit for joint pair execution. It is independent of `chunk_size`. |
| `show_progress` | Show one progress step per ISM chunk when `tqdm` is importable. Inner interpreter progress is disabled. |
| `overwrite` | Reset selected score/error columns before deciding which rows are pending. |
| `on_error` | `"raise"` aborts. `"record"` retries a failed chunk one pair at a time and stores `"<ExceptionType>: <message>"` for individually failing rows. |

The method:

- creates score columns if absent;
- scores only rows with at least one null requested score;
- always calls the interpreter with the construction-time condition,
  `center="__ism_center__"`, `retention="scalars"`, and
  interpreter `on_error="raise"`;
- updates `self.variants` after each chunk;
- returns `self` for chaining/resumption.

For `ScorerSet`, mapping keys become column names. For a scorer sequence, each
scorer's non-empty `name` is used. Names must be unique and cannot collide with
structural columns or end in `__error`.

## Query and lifecycle methods

| Method | Arguments and behavior |
| --- | --- |
| `pending(scorer_name)` | Return rows with a null scorer value. If the column does not exist, return the whole table. |
| `failed(scorer_name)` | Return rows with a non-null `<name>__error`. If no error column exists, return an empty same-schema frame. |
| `top(scorer_name, n=20, absolute=True)` | Require a known scorer and positive `n`; keep finite, non-null rows. `absolute=True` ranks by absolute effect; false ranks by descending signed score. |
| `reset_scores(*scorer_names)` | Drop named score and matching error columns. With no names, drop all current scorer/error columns. Return `self`. |
| `summary()` | Return sequence/region/center metadata, total and per-mutation counts, plus scored/pending/failed counts for every score column. |
| `write_parquet(path)` | Create parent directories, write the entire current table, and return the `Path`. |

## TF motif annotation

### `tf_annotate(motif_file, *, p_value_threshold=1e-4, peak_intervals=None)`

Scan the reference sequence in the selected ISM region on both strands. Pass
a local MEME DNA motif file with letter-probability matrices. Each retained hit
has a per-window p-value at or below the threshold. Windows containing
ambiguous bases are skipped.

Scores are log2 odds with a small background mixture. The p-value calculation
uses a zero-order background and quantizes per-position scores to 0.01 bits.
These are unadjusted tests across positions and motifs. A hit is a predicted
motif match and does not establish TF occupancy.

The method returns a Polars DataFrame and also stores it in `ism.tf_hits`.

| Columns | Meaning |
| --- | --- |
| `motif_id`, `tf` | MEME motif identifier and TF display label. |
| `start`, `end`, `start_in_region`, `end_in_region`, `strand` | Coordinates and strand in the reference sequence, plus the interval relative to the selected region. |
| `source_chrom`, `source_start`, `source_end`, `source_strand` | Genomic projection when the interval maps contiguously to one chromosome and orientation. Otherwise these fields are null. |
| `score`, `p_value`, `p_value_threshold` | Log2-odds score, approximate motif-specific p-value, and applied cutoff. |
| `overlaps_peak` | True when the hit overlaps a supplied peak interval by at least one base. Null when `peak_intervals` is omitted. |

All intervals are 0-based and half-open. Supply `peak_intervals` as reference
sequence-local `(start, end)` pairs. The overlap column labels hits without
changing which ones pass the motif threshold.

The example uses a clustered motif library with 648 matrices and 593
distinct representative TF labels. Download the file and pass its local path
to `tf_annotate()`. [H14CORE-CLUSTERED MEME file](https://github.com/WenyaWeiss/sirius2026/blob/main/motifs/H14CORE-CLUSTERED_meme_format.meme)

```python
hits = ism.tf_annotate("H14CORE-CLUSTERED_meme_format.meme")
fig, ax = ism.tf_plot(position_mode="region")
# After scoring, add the hits on the same x-axis as the variant effects.
fig, axes = ism.plot("expression_delta", show_tf=True)
```

## Plotting

### `tf_plot(*, position_mode="genomic", figsize=None)`

Plot the latest motif hits as labeled intervals in strand-specific colors.
Returns `(figure, axis)`. The default height grows with the number of TF
labels. Use `position_mode="region"` or `"sequence"` when the reference region
does not map contiguously to one genomic interval.

### `plot(...)`

```python
plot(
    scorer_name,
    *,
    mutation_type="substitution",
    target_col=None,
    position_mode="region",
    figsize=None,
    show_tf=False,
)
```

| Argument | Purpose |
| --- | --- |
| `scorer_name` | Existing score column with at least one finite value. |
| `mutation_type` | `"substitution"`, `"deletion"`, or `"all"`. The filtered table must be non-empty. |
| `target_col` | Optional existing numeric table column plotted as a measured-effect panel above the model score. |
| `position_mode` | `"region"` uses `position_in_region`. `"sequence"` uses local sequence `position`. `"genomic"` uses source coordinates and requires a contiguous genomic projection of the region. |
| `figsize` | Matplotlib figure size. `None` sizes the figure for the selected panels and TF labels. |
| `show_tf` | After `tf_annotate()`, add a motif panel below the scores with the same x-axis. |

The plot uses ALT bases as colors, REF bases as marker shapes, and returns
`(figure, axes_array)`.

### `position_attributions(scorer_name)`

Return a Polars DataFrame with one row per position in the ISM region. Columns
are `position`, `position_in_region`, `ref`, `n_scored`, and `attribution`. The
score column should contain ALT-minus-REF effects. Attribution is `-sum(delta) / 4`
because the unchanged reference contributes a zero effect to the four-base mean.
A positive value means the three substitutions lower the predicted score on
average. Positions with fewer than three finite SNV scores have null attribution.

### `rank_tf_hits(...)`

After `tf_annotate()`, rank scanned motif hits by their agreement with a selected
ISM scorer. The sequence p-value threshold used for the scan sets the candidate
pool. To include weaker sequence matches, choose the candidate cutoff when
calling `tf_annotate()`.

```python
ranked = ism.rank_tf_hits(
    "expression_delta", min_profile_similarity=0.4, min_peak_coverage=0.5
)
```

Peaks are contiguous bases selected by the same `attribution_threshold` and
`attribution_threshold_mode` used in the logo plot. At each overlapping base,
the method centers the four ISM
effects (including zero for REF) and the strand-aligned motif log-odds weights
across A/C/G/T. It compares these profiles with cosine similarity weighted by
absolute attribution. Sites need all three finite SNV scores at a base for
that base to contribute.

The returned Polars table keeps the scan columns and adds
`peak_start`, `peak_end` (sequence-local, zero-based, half-open),
`peak_coverage` (the fraction of the peak's absolute attribution covered by
the hit), signed `profile_similarity` (-1 to 1), and one-based `peak_rank`.
Each row is one hit at one peak. Profile agreement ranks hits within each peak,
with the sequence p-value breaking ties. The top hit from each peak is listed
before second-ranked hits.

Set `max_overlapping_tfs=5` to allow at most five motif hits over any DNA base.
The cap considers each hit's full footprint in priority order across all peaks.
A hit shared by several peaks counts once, and `peak_rank` records its rank
before the overlap cap is applied.

`profile_direction="positive"` looks for agreement between motif affinity and
model output. Use `"negative"` for inverse agreement or `"either"` to rank
by the magnitude of the similarity while retaining its sign. Both cutoffs are
between 0 and 1. The profile cutoff measures effect agreement on the chosen
scorer, so tune it for the scorer and data.

### `plot_attribution(...)`

Install the optional logo renderer from the API directory with
`python -m pip install '.[logos]'`. After scoring the substitutions, request the motif panel with `show_tf=True`.

```python
ism.tf_annotate("H14CORE-CLUSTERED_meme_format.meme")
fig, axes = ism.plot_attribution("expression_delta", show_tf=True)
```

The upper panel draws each reference letter with height equal to its
attribution magnitude and places it above or below zero according to the sign.
With `show_tf=True`, a lower panel draws motif logos in bits against a uniform
DNA background. The panels share their position axis. On a reverse-strand
genomic projection, the letters and motifs are oriented to the genomic axis.

- `attribution_threshold` is a quantile of finite absolute attribution.
  For example, `0.7` highlights nonzero bases at or above the 70th percentile,
  and `None` uses the 90th percentile. Set
  `attribution_threshold_mode="absolute"` with a numeric threshold to use a
  fixed cutoff in score units. Adjacent highlighted bases share a filled frame.
- The motif panel shows hits that overlap at least one highlighted base and
  shades the corresponding part of each motif. `max_motif_hits=12` limits the
  display to hits with the strongest overlapping attribution, using the
  motif p-value to break ties.
- With `show_tf=True`, set `min_profile_similarity`, `min_peak_coverage`, or
  a nondefault `profile_direction` to use `rank_tf_hits()` for motif selection.
  Repeated hits are drawn once.
- `max_overlapping_tfs` caps how many displayed motif hits cover each base.
  It applies before `max_motif_hits`, which caps the total number displayed.
  Use `None` for unlimited overlap or `0` to show zero motif hits.
- `position_mode` and `figsize` follow the other plots. The method returns
  a Matplotlib `(figure, axes_array)`.

### `plot_substitution_matrix(...)`

```python
plot_substitution_matrix(
    scorer_name,
    *,
    cmap="coolwarm",
    center=0.0,
    figsize=(16.0, 4.0),
)
```

Builds a four-row A/C/G/T-by-region-position matrix from available substitution
scores.

| Argument | Purpose |
| --- | --- |
| `cmap` | Matplotlib colormap name. |
| `center` | If not `None`, choose symmetric color limits around this value. `None` lets Matplotlib choose limits. |
| `figsize` | Figure dimensions. |

Unsampled substitutions remain masked. At least one finite substitution score
is required. Returns `(figure, axes)`.

## Typical lifecycle

```python
from gena_expression import ISM

ism = ISM(
    sequence,
    center="tss",
    region="enhancer",
    condition=condition,
    n_substitutions=None,
    n_deletions=100,
    random_seed=7,
)

ism.score(
    interpreter,
    scorer,
    chunk_size=128,
    max_records_per_forward=64,
)

strongest = ism.top(scorer.name, n=20)
ism.write_parquet("results/ism.parquet")
fig, ax = ism.plot_substitution_matrix(scorer.name)
```

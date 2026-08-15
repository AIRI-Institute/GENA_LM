# `gena_expression` package

`gena_expression` is a Python API for constructing annotated DNA inputs, running
sequence-and-condition inference, building reference/alternative allele pairs,
and reducing model predictions to expression or coordinate-aware track scores.

This README documents the modules stored directly in this package directory and
acts as an index to the focused documentation in each subpackage. It does not
duplicate the detailed class references in those subpackage READMEs.

## Package map

| Path | Responsibility | What its README covers |
| --- | --- | --- |
| [`dna/`](dna/README.md) | FASTA/GTF-backed genome access, genomic windows, GenBank plasmids, MPRA insertion, and context builders | `Context`, `Genome`, `GenomeContext`, `GenomeRegion`, `PlasmidRecord`, `PlasmidCollection`, `PlasmidContext`, related value classes, and primer/context utilities |
| [`inference/`](inference/README.md) | Model loading, centered tokenization, batching, prediction, and prediction retention | `SequenceModel`, `CenteredTokenizer`, `TokenizedSequence`, `Prediction`, `ExpressionPrediction`, `PairPrediction`, grouping modes, pair execution, and preprocessing arguments |
| [`scoring/`](scoring/README.md) | Pair interpretation, absolute and pair-effect scorers, result containers, plotting, and result retention | `VariantInterpreter`, every scorer, `ScoringResult`, report classes, scorer arguments, aggregation rules, and retained-data requirements |
| [`ism/`](ism/README.md) | In-silico substitution/deletion library generation, resumable scoring, tables, and plots | `ISM`, its mutation-table schema, constructor controls, scoring lifecycle, query/export methods, and plotting arguments |
| [`optimization/`](optimization/README.md) | Evolutionary optimization of a fixed-length sequence region | `SequenceOptimizer`, generator contracts, target scoring, fitness, selection strategies, public tables, and run/export methods |
| `conditions.py` | Condition text and deterministic JSON metadata lookup | Documented below |
| `config.py` | Context-local prediction/scoring retention policy | Documented below |
| `sequences.py` | Features, annotated sequences, coordinate maps, and allele pairs | Documented below |
| `tracks.py` | Interval-level track containers, bigWig conversion, and plotting | Documented below |
| `variants.py` | Variant parsing, validation, application, and pair construction | Documented below |
| `__init__.py` | Curated package-level import surface | See [Public imports](#public-imports) |

Generated `__pycache__` directories contain interpreter bytecode and are not
source or documentation folders.

## End-to-end object flow

The main object flow is:

1. `Condition` supplies the natural-language model condition.
2. `AnnotatedSequence` holds DNA, interval features, coordinate provenance, and
   metadata.
3. `Variant` and a context builder produce a `SequencePair`.
4. `SequenceModel` turns one sequence or a pair into prediction objects.
5. A scorer turns a prediction into a score-bearing result.
6. `VariantInterpreter` combines pair construction, prediction, scoring, and
   optional retention compaction.
7. `ISM` and `SequenceOptimizer` build higher-level workflows on those same
   sequence, model, and scoring contracts.

## Core conventions

### Coordinates

- `Feature`, `AnnotatedSequence`, `CoordinateMap`, `SequencePair`, scorer
  windows, ISM regions, and `Variant.pos` use 0-based, half-open sequence
  coordinates unless a method explicitly says otherwise.
- A half-open interval `[start, end)` includes `start` and excludes `end`.
- `Genome.fetch()`, `Genome.sequence()`, and `Genome.features()` also accept
  0-based, half-open genomic intervals.
- GTF/GFF input is converted from 1-based inclusive coordinates to 0-based
  half-open coordinates.
- `SafeHarborSite` and `GenomeInterval` store 1-based inclusive coordinates.
- `Variant.from_str(..., coordinate_system="auto")` treats the parsed string
  position as 1-based and subtracts one. For mapping/object rows accepted by
  `VariantInterpreter`, only the explicit `"1-based"` mode subtracts one.

### Copy-returning domain objects

`Feature`, `CoordinateSegment`, `CoordinateMap`, `AnnotatedSequence`,
`SequencePair`, and `Variant` are frozen dataclasses. Operations such as
`AnnotatedSequence.add_feature()`, `replace()`, and `reverse_complement()` return
new objects rather than mutating the source.

`Track`, prediction containers, and scoring-result containers are mutable
dataclasses. In particular, `VariantInterpreter` prefixes result names after a
scorer returns them.

### Named feature resolution

Feature lookup is method-specific:

- `AnnotatedSequence.feature(name)` returns the first exact feature-name or
  feature-type match.
- `AnnotatedSequence.select("text")`, `update_feature("text")`, and
  `remove_feature("text")` require a unique substring match against feature
  names or types.
- `ISM` and `SequenceOptimizer` require a unique exact name/type match, with an
  optional `feature:` prefix.
- Track feature scorers match exact names or types and may deliberately use all
  matching features.

### Optional dependencies

Imports are generally local to the method that needs them. Relevant optional
packages include:

| Capability | Dependency used by the code |
| --- | --- |
| Model execution/loading | PyTorch, Transformers, Hydra, Safetensors |
| FASTA, GTF/GFF, tabix | `pysam`; Polars may be used to sort an unindexed annotation |
| DataFrames | pandas or Polars, depending on the method |
| Plotting | Matplotlib; Seaborn is not required by these package modules |
| bigWig input | `pyBigWig` |
| Regression model loading | joblib; scoring also imports NumPy |

The parent project metadata declares Python 3.10 or newer and exposes optional
dependency groups. Consult the subpackage README before using an optional
workflow.

## Public imports

The package root re-exports the following names from `gena_expression`:

| Domain | Public names |
| --- | --- |
| Conditions | `Condition`, `DescriptionLookup` |
| Retention | `PredictionRetention`, `ScoringRetention`, `RetentionPolicy`, `RetentionMode`, `RetainedDataError`, `FULL_RETENTION`, `SCALAR_RETENTION`, `get_retention_policy`, `set_retention_mode`, `set_retention_policy`, `retention_mode` |
| Sequences and variants | `Feature`, `SourceCoordinate`, `CoordinateSegment`, `CoordinateMap`, `AnnotatedSequence`, `SequencePair`, `Variant` |
| Tracks | `Track`, `TrackPrediction` |
| dna | `Context`, `Genome`, `GenomeContext`, `GenomeInterval`, `GenomeRegion`, `SafeHarborSite`, `FeatureMatch`, `PlasmidMetadata`, `PlasmidRecord`, `PlasmidCollection`, `PlasmidContext`, `infer_table18_primer_tails` |
| Inference | `CenteredTokenizer`, `TokenizedSequence`, `SequenceModel`, `Prediction`, `ExpressionPrediction`, `PairPrediction` |
| Scoring | `VariantInterpreter`, `ExpressionScorer`, `ExpressionDeltaScorer`, `TokenWindowScorer`, `TrackWindowScorer`, `TrackEffectPeakScorer`, `TrackFeatureScorer`, `TrackAllFeaturesScorer`, `TrackFeatureBuilder`, `RegressionScorer`, `ScorerSet`, `ScoreWindow`, `DisplayWindow`, `ResultIdentity`, `PredictionScoringResult`, `ScoringResult`, `PredictionReport`, `VariantReport` |
| Workflows | `ISM`, `SequenceOptimizer` |

Some module-level helpers and type aliases are intentionally available only from
their defining module or subpackage. Their folder READMEs identify them.

---

# Root-module reference

## `conditions.py`

### `Condition`

`Condition` is the structured condition attached to every prediction. The
inference layer converts strings and mappings to a `Condition`; prediction and
scoring results retain it or its name depending on the retention policy.

Constructor:

```python
Condition(
    name,
    description,
    assay=None,
    cell_type=None,
    tissue=None,
    metadata=None,
)
```

| Argument | Meaning |
| --- | --- |
| `name: str` | Human-readable identity used in result tables, labels, grouping metadata, and `condition_name`. |
| `description: str \| Mapping[str, Any]` | Exact text, or ordered metadata used to generate the text passed to the description tokenizer. |
| `assay: str \| None` | Optional assay label carried as metadata. It does not alter `text()` by itself. |
| `cell_type: str \| None` | Optional cell-type label carried as metadata. |
| `tissue: str \| None` | Optional tissue label carried as metadata. |
| `metadata: Mapping \| None` | Additional user metadata serialized by `to_dict()`. |

Methods:

| Method | Arguments and result | Used elsewhere |
| --- | --- | --- |
| `set_description_formatter(formatter)` | Configure a callable or `"/path/file.py::ClassName::static_method"` for this Python runtime. `None` clears it. | Models use this runtime override when no per-model formatter is supplied. |
| `clear_description_formatter()` | Clear the runtime formatter. | Structured descriptions then fail until a formatter is configured. |
| `get_description_formatter()` | Return the resolved runtime formatter, or `None`. | `SequenceModel` snapshots it during construction. |
| `text()` | Returns `description` unchanged when it is a string; otherwise uses the configured runtime formatter. | Standalone condition rendering. Model inference uses the model's snapshotted formatter. |
| `to_dict()` | Returns all constructor fields in a JSON-friendly dictionary; `metadata=None` becomes `{}`. | `Prediction.to_dict()` includes it. |

### `metadata_to_description(meta)`

Converts each mapping entry, in mapping iteration order, to a sentence of the
form `"<clean key> is <clean value>."`. Underscores become spaces, quotes are
removed from values, square brackets are removed from keys, and several
leading metadata prefixes are stripped. This function remains available for
explicit use, but is not installed as an implicit default.

Standalone `Condition.text()` calls with structured descriptions require runtime
configuration:

```python
Condition.set_description_formatter(my_callable)

Condition.set_description_formatter(
    "/path/to/dataset_file.py::DatasetDescriptions::make_description_from_json"
)
```

For model inference, a formatter can instead be supplied directly to
`SequenceModel`. When `SequenceModel.load()` receives neither a per-model nor a
runtime formatter, it loads `make_description_from_json` from the dataset class
selected by the model config. Formatter callables always receive a built-in
`dict` first. Every additional fixed positional or keyword-only parameter
receives `None`; variadic parameters do not receive invented values. The
callable must return `str`.

### `DescriptionLookup`

`DescriptionLookup` indexes `*.json` files in one directory and chooses one
candidate per requested key deterministically.

Constructor:

```python
DescriptionLookup(
    json_dir,
    keys,
    seed=0,
    strict=False,
    search_fields=None,
    **filters,
)
```

| Argument | Meaning |
| --- | --- |
| `json_dir: str \| Path` | Existing directory whose immediate `*.json` files are read in sorted path order. Subdirectories are not scanned. |
| `keys: Iterable[str]` | Requested lookup labels. Each key is searched case-insensitively in candidate JSON content. |
| `seed: int \| str` | Included with the key in a SHA-256 digest used to select a stable candidate. It does not use process-random hash state. |
| `strict: bool` | If `True`, construction raises `KeyError` for a key with no candidate. If `False`, unmatched keys are omitted from the selected lookup but remain queryable through `candidates()`. |
| `search_fields: Sequence[str] \| None` | `None` searches the serialized full JSON record. Otherwise, only the named dotted paths are searched. |
| `filters=...` inside `**filters` | Mapping of dotted JSON fields to values. All are exact global filters applied before text matching. |
| `key_filters=...` inside `**filters` | Mapping from requested key to an exact-filter mapping used only for that key. |
| Other keyword filters | Exact global dotted-field filters. A keyword whose name is also one of `keys` and whose value is a mapping becomes a key-specific filter. |

Construction raises `FileNotFoundError` if `json_dir` is absent and `ValueError`
if no JSON record survives the global filters.

Public methods:

| Method | Purpose |
| --- | --- |
| `lookup[key]` | Return the selected metadata mapping; standard `KeyError` applies to an unmatched key. |
| `key in lookup` | Test whether a selection exists. |
| `keys()`, `items()`, `values()` | Return views over selected matches only. |
| `candidates(key)` | Return all candidate file paths considered for the key, or an empty list for an unknown/unmatched key. |
| `condition(key, assay=None)` | Build `Condition(name=key, description=selected_metadata, assay=assay, cell_type=key, metadata={"json_dir": ...})`. This result can be passed directly to `SequenceModel`. |

## `config.py`

Retention controls what remains on returned predictions and scoring results. It
does not change the full data supplied to scorers: `VariantInterpreter` predicts
with full retention, scores, and then compacts each returned result.

### `RetainedDataError`

Raised when a method needs a field removed by retention—for example,
`Prediction.track()` without retained tokens/logits or `ScoringResult.plot_*()`
without retained predictions/tracks.

### `RetentionMode`

String enum with two values:

- `RetentionMode.FULL` / `"full"`
- `RetentionMode.SCALARS` / `"scalars"`

### `PredictionRetention`

```python
PredictionRetention(
    sequence=True,
    logits=True,
    outputs="full",
    tokens=True,
    description_tokens=True,
    provenance=True,
)
```

| Field | Effect |
| --- | --- |
| `sequence` | Retain the input `AnnotatedSequence`; pair predictions retain their `SequencePair` only when this is `True`. |
| `logits` | Retain the dedicated logits field and the `"logits"` output alias. |
| `outputs` | `"full"` keeps output objects; `"scalars"` converts outputs to plain Python numbers. |
| `tokens` | Retain `TokenizedSequence`, required for `Prediction.track()`. |
| `description_tokens` | Retain the encoded condition payload. |
| `provenance` | Retain batch/model provenance. |

Only `"full"` and `"scalars"` are valid for `outputs`.

### `ScoringRetention`

```python
ScoringRetention(
    prediction=True,
    score_window=True,
    tracks="all",
    features="full",
    warnings=True,
    provenance=True,
)
```

| Field | Effect |
| --- | --- |
| `prediction` | Retain the `PairPrediction` on each `ScoringResult`. |
| `score_window` | Retain `score_window`, `ref_score_windows`, and `alt_score_windows`. |
| `tracks` | `"all"` keeps reference, alternative, and delta rows; `"delta"` keeps only delta rows; `"none"` drops all. |
| `features` | `"full"` keeps all feature metadata; `"scores"` keeps only each entry's `key` and `score`; `"none"` drops features. |
| `warnings` | Retain scorer warnings. |
| `provenance` | Retain scorer provenance. |

### `RetentionPolicy`

Groups `prediction: PredictionRetention` and `scoring: ScoringRetention`.

The constants are:

- `FULL_RETENTION`: all defaults above.
- `SCALAR_RETENTION`: drops sequences, logits, tokens, description tokens,
  prediction/scoring provenance, pair predictions, and tracks; converts outputs
  to scalars; retains score windows, warnings, compact feature scores, and a
  lightweight `ResultIdentity`.

### Retention functions

| Function | Arguments and effect |
| --- | --- |
| `get_retention_policy()` | Return the current context-local policy. |
| `set_retention_mode(mode)` | Resolve `"full"` or `"scalars"`, set it for the current execution context, and return the policy. |
| `set_retention_policy(policy)` | Require a `RetentionPolicy`, set it, and return it. |
| `retention_mode(mode)` | Context manager that temporarily installs a preset/policy and restores the previous one. |
| `retention_policy_for(value)` | Module-level normalizer for `None`, a policy, enum, or preset string. `None` means the current context policy. It is used throughout inference and scoring but is not re-exported at package root. |

## `sequences.py`

### `Feature`

Immutable interval annotation:

```python
Feature(
    name,
    start,
    end,
    type="feature",
    strand=None,
    source=None,
    metadata=None,
)
```

`start` must be non-negative and `end >= start`. Zero-length features are
allowed.

| Method | Arguments and result |
| --- | --- |
| `length()` | Return `end - start`. |
| `copy(**changes)` | Return a dataclass copy with changed fields. |
| `shift(offset)` | Add `offset` to `start` and `end`. |
| `clip(start, end)` | Intersect with `[start, end)` and express the result relative to `start`; return `None` when there is no positive-width overlap. |
| `overlaps(start, end)` | Test half-open interval overlap. |
| `contains(position)` | Test `start <= position < end`. |
| `to_dict()` | Serialize all fields; missing metadata becomes `{}`. |

`Feature` is used by context builders, tokenization, variants, ISM,
optimization, coordinate-aware scorers, and annotation plots.

### `SourceCoordinate`

Immutable return value from `CoordinateMap.seq_to_source()`:

| Field | Meaning |
| --- | --- |
| `source` | Source label such as `"sequence"` or `"genome"`. |
| `position` | Mapped source position, or `None` when a segment has no numeric source span. |
| `chrom` | Optional chromosome/source-contig label. |
| `strand` | Optional source orientation. |
| `metadata` | Segment metadata. |

### `CoordinateSegment`

Describes one contiguous mapping from `[seq_start, seq_end)` to a source:
`source`, optional `chrom`, optional `[source_start, source_end)`, optional
`strand`, and metadata. `shifted(offset)` moves only the sequence-side
coordinates and is used while concatenating annotated sequences.

### `CoordinateMap`

Constructor:

```python
CoordinateMap(segments=())
```

The segment sequence is frozen as a tuple.

| Method | Arguments and result | Used elsewhere |
| --- | --- | --- |
| `from_length(length, source="sequence")` | Create one identity-like segment, or an empty map when `length <= 0`. | Default map created by `AnnotatedSequence`. |
| `seq_to_source(position)` | Return the first covering segment as a `SourceCoordinate`; honor reverse-strand mapping. | ISM stores source coordinates in its table. |
| `source_to_seq(source, position, chrom=None)` | Map a source position into sequence coordinates; optionally require a chromosome. |
| `slice(start, end)` | Clip segments to a sequence slice, shift them to zero, and adjust source spans. | `AnnotatedSequence.__getitem__()`. |
| `reverse_complement(sequence_length)` | Reverse segment order/positions and flip `+`/`-` strands. | `AnnotatedSequence.reverse_complement()`. |
| `concat(other, offset)` | Append `other` after shifting its sequence coordinates. | `AnnotatedSequence.concat()`. |
| `to_frame()` | Return a pandas DataFrame with one row per segment. |

### `AnnotatedSequence`

Central immutable DNA container:

```python
AnnotatedSequence(
    sequence,
    name=None,
    features=(),
    coordinate_map=None,
    metadata=None,
)
```

Construction uppercases `sequence`, freezes `features` as a tuple, creates
`CoordinateMap.from_length(len(sequence))` when no map is supplied, and copies
metadata to a plain dictionary. The constructor does not validate the DNA
alphabet.

Basic access:

| Method | Behavior |
| --- | --- |
| `str(sequence)` | Return the DNA string. |
| `len(sequence)` | Return base length. |
| `sequence[index]` | Return one base. |
| `sequence[slice]` | For step `1`, return an annotation- and coordinate-map-preserving `AnnotatedSequence`; for other steps, return a plain string. |

Feature methods:

| Method | Important arguments |
| --- | --- |
| `add_feature(name, start, end, type="feature", strand=None, source=None, metadata=None)` | Append a newly constructed `Feature`. |
| `feature(name, required=True)` | Return the first exact `Feature.name == name` or `Feature.type == name`; raise `KeyError` when required and absent. |
| `features_at(position)` | Return every containing feature. |
| `features_overlapping(start, end)` | Return every overlapping feature. Tokenization uses this to annotate token records. |
| `update_feature(feature, **changes)` | Resolve one `Feature` object or unique substring string and replace it with `Feature.copy(**changes)`. |
| `update_features(where, **changes)` | Apply changes to every feature for which `where(feature)` is true. |
| `remove_feature(feature)` | Resolve and remove exactly one feature. |
| `remove_features(where=None, *, name=None, type=None, source=None)` | Remove features matching all supplied filters. At least one filter/callable is required. |
| `keep_features(where=None, *, name=None, type=None, source=None)` | Keep only features matching all supplied filters. At least one is required. |
| `map_features(fn)` | Map every feature; return values must be `Feature` or `None`, where `None` drops the feature. |

Selection and transformation methods:

| Method | Arguments and result | Used elsewhere |
| --- | --- | --- |
| `select(start, end, strand=None, *, feature=None, features=None, name=None)` | Select coordinates; a unique feature substring; or ordered feature selectors that are concatenated. `strand="-"` reverse-complements the selected interval. If no explicit strand is supplied for feature selection, a feature's `+`/`-` strand is used, otherwise `+`. | Optimization/plotting code uses selection semantics; it is also the general annotation-preserving extraction method. |
| `resolve_center(center)` | Integers pass through. A `Feature` or exact named feature resolves to its midpoint. A string may start with `feature:`. | `CenteredTokenizer` and pair prediction use it. |
| `window(center, size, name=None, pad="N")` | Return exactly `size` bases centered on the resolved center; add left/right padding features when outside the sequence. `pad` is repeated and is intended to be a one-character fill. | Useful before tokenization and context inspection. |
| `reverse_complement(name=None)` | Reverse-complement DNA; reverse/remap features and coordinate segments; flip `+`/`-` strands; add metadata flag. | Genome minus-strand extraction and sequence selection. |
| `AnnotatedSequence.concat(*parts, name=None)` | Concatenate DNA, shift features/maps, and return a new sequence. Metadata from parts is not merged. | Genome/plasmid context assembly and primer-defined variants. |
| `replace(start, end, replacement, preserve_partial_features=False)` | Replace `[start, end)` with a string or annotated replacement. Shift later features; remove overlapping features by default; optionally keep surviving pieces. Annotated replacement features are inserted; a plain string receives a generic `"replacement"` feature. | `Variant.apply_to()` uses `preserve_partial_features=True`; plasmid MCS replacement also uses it. |
| `with_metadata(**metadata)` | Merge key/value pairs into copied metadata. |

Export and visualization:

| Method | Arguments and result |
| --- | --- |
| `to_fasta(header=None)` | Return FASTA text wrapped at 80 bases; use `header`, then `name`, then `"sequence"`. |
| `to_dict()` | Serialize name, DNA, features, and metadata. The coordinate map is not included. |
| `feature_frame()` | Return a pandas DataFrame of feature dictionaries. |
| `plot_annotations(start=None, end=None, types=None, hide_types=("source",), title=None, ax=None, figsize=(12, 2.8), label_features=True, show_legend=True, save_path=None)` | Draw interval annotations with Matplotlib. `types` is an inclusion filter; `hide_types` is an exclusion filter; `ax` allows composition; `save_path` writes at 180 dpi. Returns `(figure, axes)`. |

### `SequencePair`

Immutable reference/alternative container:

```python
SequencePair(ref, alt, variant=None, metadata=None)
```

Both alleles must be `AnnotatedSequence` objects by contract. `metadata` is
copied to a dictionary.

| Method | Arguments and result | Used elsewhere |
| --- | --- | --- |
| `variant_feature()` | Return the first exact `"variant"` feature from REF, then ALT. | Track window and peak-center resolution. |
| `variant_features()` | Return `(ref_feature, alt_feature)`. | Difference/mapping logic. |
| `assert_compatible()` | Currently checks only that both sequences are non-empty. | Context, variant, and ISM pair builders call it. |
| `map(fn)` | Apply the same `AnnotatedSequence -> AnnotatedSequence` transform to both alleles; preserve pair variant/metadata; validate return types. |
| `coordinate_mapper()` | Infer the exact changed intervals and return the internal allele-coordinate mapper. | Coordinate-aware track scorers cache/use it for indels. |
| `map_ref_interval_to_alt(start, end)` | Map a REF-local half-open interval into ALT coordinates. |
| `map_alt_interval_to_ref(start, end)` | Map an ALT-local half-open interval into REF coordinates. |
| `difference_interval(method="auto")` | Return `(ref_start, ref_end, alt_start, alt_end)`. Valid methods are `"auto"`, `"feature"`, and `"string"`. `"auto"` prefers variant features; `"feature"` requires one; `"string"` uses shared prefix/suffix comparison. |
| `plot_difference(flank=40, method="auto", show_gaps=True, gap_char="-", ax=None, figsize=(12, 2.4), title=None, save_path=None)` | Plot the local allele change. `gap_char` must be one visible character. Gaps are display-only. Returns `(figure, axes)`. |
| `to_dict()` | Serialize both sequences, variant when it has `to_dict()`, and metadata. |

The internal `_PairCoordinateMapper` is not exported. It linearly maps
boundaries inside a replacement and shifts coordinates before/after it.

## `variants.py`

### `Variant`

Immutable substitution, insertion, deletion, or allele-replacement descriptor:

```python
Variant(
    chrom,
    pos,
    ref,
    alt,
    id=None,
    strand="+",
    metadata=None,
)
```

| Argument | Meaning |
| --- | --- |
| `chrom: str \| None` | Reference contig. Use `None` for sequence-only variants. |
| `pos: int \| None` | 0-based reference start. Non-`None` values must be non-negative. |
| `ref`, `alt` | Alleles. Construction uppercases them and converts exactly `"-"` to the empty allele. |
| `id` | Optional stable label used by pair/result naming. |
| `strand` | Stored on variant-derived features; no constructor validation is applied. |
| `metadata` | Copied dictionary used by sequence-only and primer/plasmid workflows. |

Construction helpers:

| Method | Important arguments |
| --- | --- |
| `from_str(text, genome=None, coordinate_system="auto")` | Requires four fields after whitespace removal and `>` to `:` replacement: `chrom:pos:ref:alt`. A range-like position uses the text before the first `-`. `"auto"` and `"1-based"` subtract one; `"0-based"` does not. If `genome` is supplied, call `validate()`. |
| `from_vcf_record(record, genome=None)` | Read `chrom/CHROM`, 1-based `pos/POS`, `ref/REF`, first `alts` item or `ALT`, and optional `id/ID`; subtract one from position. |
| `from_sequences(ref, alt, name=None)` | Find the minimal changed interval by shared prefix/suffix comparison. Annotated inputs contribute serialized features/names to metadata so downstream plasmid construction can restore them. |

Inspection/application:

| Method | Purpose | Used elsewhere |
| --- | --- | --- |
| `validate(genome)` | Fetch the declared REF span and emit `RuntimeWarning` on mismatch; sequence-only variants raise. | `from_str()`, `from_vcf_record()`, and `VariantInterpreter` row normalization. |
| `as_feature(name="variant")` | Build a one-base visual anchor for an empty REF, otherwise a REF-length variant feature. |
| `length_change()` | Return `len(alt) - len(ref)`. |
| `is_snv()` | True only for unequal one-base REF/ALT. |
| `is_indel()` | True when allele lengths differ. |
| `apply_to(sequence, offset=0)` | Replace at local `offset`, warn on REF mismatch, preserve surviving feature pieces, and add an ALT-side variant feature. | `GenomeContext` and ISM construct ALT sequences with it. |
| `to_dict()` | Serialize all fields. |
| `plot(ref_sequence=None, alt_sequence=None, flank=40, ax=None, figsize=(12, 2.4), save_path=None)` | With both context sequences, delegate to `SequencePair.plot_difference()`; otherwise draw the allele-level change. |

Pair materialization:

```python
variant.to_sequence_pair(
    context=None,
    *,
    genome=None,
    name=None,
    window_bp=None,
    primer_pairs=None,
    primer_locations=None,
    primer_window_bp=None,
    primer_flank_bp=100,
    primer_location_padding_bp=0,
    primer_min_match=12,
    primer_edge_slop=25,
    include_primer_tails=False,
    primer_pair_name=None,
)
```

Modes are selected in this order:

1. `primer_pairs` present: require `genome`, `chrom`, and `pos`; infer a
   primer-defined fragment. `context` is ignored.
2. `window_bp` present: construct `GenomeContext(length=window_bp)` and build a
   centered genomic pair.
3. Otherwise: require `context` and call
   `context.build(variant, genome=genome, name=name)`.

Primer-mode arguments:

| Argument | Use |
| --- | --- |
| `primer_pairs` | Mapping of pair name to `(forward_primer, reverse_primer)`, or an iterable of pairs. |
| `primer_locations` | Optional mapping to `(chromosome, 1-based start, 1-based end)` intervals used to select/search known constructs. |
| `primer_window_bp` | Variant-centered fallback search width when location intervals are not used. |
| `primer_flank_bp` | Extra flanking bases around a known location interval. |
| `primer_location_padding_bp` | Global integer padding or per-location mapping. |
| `primer_min_match` | Minimum primer-suffix annealing length. |
| `primer_edge_slop` | Preferred distance from the expected construct edge. |
| `include_primer_tails` | If true, include forward cloning tail and reverse-complemented reverse-primer tail as annotated scars. |
| `primer_pair_name` | Restrict matching to one named primer pair. |

`VariantInterpreter.score_variant()` and `score_variants()` forward extra
keyword arguments to this method.

## `tracks.py`

### `Track`

Mutable interval-valued track:

```python
Track(name, values, tokens, channel=0)
```

`values[i]` corresponds to `tokens[i]`. Token rows are expected to include
`start` and `end`; the constructor does not validate equal lengths.

| Method | Arguments and behavior | Used elsewhere |
| --- | --- | --- |
| `from_bw(path, *, chrom, start=None, end=None, center=None, size=None, strand="+", name=None, channel=None, missing_value=nan)` | Load a 1-bp track with `pyBigWig`. Provide exactly `start/end` or `center/size`; coordinates must be within the chromosome. Minus strand reverses values and genomic positions while output `start/end` remain local 0-based coordinates. |
| `rescale_to_1bp(name=None, fill_value=None)` | Sort interval rows and repeat each value once per covered base. Overlapping or non-positive intervals raise. If `fill_value` is supplied, fill gaps between intervals. |
| `to_1bp(...)` | Alias for `rescale_to_1bp()`. |
| `to_frame()` | Return a pandas DataFrame combining each token row with a value column named `self.name`. |
| `plot(ax=None, figsize=(12, 3), title=None, ylabel=None, color="tab:blue", color_by_sign=True, save_path=None)` | Draw interval values with Matplotlib. Sign coloring uses green/red/gray; with `color_by_sign=False`, use the supplied `color`. Returns `(figure, axes)`. |

### `TrackPrediction`

Subclass of `Track` retained as the prediction-track name. It adds no fields or
methods. `Prediction.track()` and `PairPrediction.delta_track()` construct it,
and coordinate-aware scorers consume its values/token rows.

## Minimal composition example

```python
import gena_expression as gx

condition = gx.Condition(
    name="example",
    description={"cell_type": "example", "assay": "RNA-seq"},
)

reference = gx.AnnotatedSequence(
    "A" * 100,
    name="reference",
    features=(gx.Feature("tss", 50, 51, type="tss"),),
)
variant = gx.Variant(chrom=None, pos=50, ref="A", alt="G", id="example_variant")
alternative = variant.apply_to(reference, offset=50)
pair = gx.SequencePair(reference, alternative, variant=variant)
```

Continue with [`inference/README.md`](inference/README.md) for prediction or
[`scoring/README.md`](scoring/README.md) for interpretation.

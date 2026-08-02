# Scoring and interpretation

The `gena_expression.scoring` subpackage turns model predictions into absolute
sequence scores or reference/alternative effects. It also supplies the
high-level `VariantInterpreter`, result/report containers, result views,
plotting, JSON/table export, and scoring-result retention.

Read the package [README](../README.md) for coordinates, sequence/variant
classes, and retention configuration. Read
[`inference/README.md`](../inference/README.md) for grouping, pair execution,
CPU preprocessing, and prediction-container details.

## Module map

| Module | Contents |
| --- | --- |
| `interpreter.py` | `VariantInterpreter` |
| `scorers.py` | Scorer protocol, expression/track/regression scorers, and `ScorerSet` |
| `results.py` | Score/display windows, result/report classes, exports, plots, and retention helpers |
| `__init__.py` | Public scoring exports |

## Scorer contracts

Pair scorers satisfy the structural protocol:

```python
name: str

def score(
    prediction: PairPrediction,
) -> ScoringResult | VariantReport:
    ...
```

No inheritance is required. `VariantInterpreter` invokes `score()` after full
REF/ALT predictions have been generated.

Absolute single-sequence scorers expose:

```python
def score_prediction(
    prediction: Prediction,
) -> PredictionScoringResult:
    ...
```

`ExpressionScorer`, `TrackWindowScorer`, `TrackFeatureScorer`, and
`TrackAllFeaturesScorer` implement this method. `SequenceOptimizer` requires it.

## Shared scorer arguments

| Argument | Meaning |
| --- | --- |
| `name` | Result name before `VariantInterpreter` prefixes the input label. It is also used as the default ISM table-column name. |
| `sign` | `"alt-ref"` is ALT minus REF; `"ref-alt"` negates that pair effect. It has no effect on absolute `score_prediction()` methods. |
| `track` | Label passed to `Prediction.track(name=...)`. The current prediction implementation uses the same logits tensor regardless of this label. |
| `channel` | Integer selects one logit channel; `None` averages channels when logits are 3-D and have more than one channel. |
| `center` | Integer local sequence coordinate, `Feature` midpoint, or feature name. Track scorers also accept `feature:<name>`. |
| `aggregate` | How track values overlapping a window are reduced. Exact choices depend on the scorer. |

Coordinate-aware track reductions use base-pair overlap:

- `"sum"`, `"auc"`, and `"weighted_sum"` all return
  `sum(value * covered_bases)`.
- `"mean"` divides that weighted area by covered bases.
- `"max"` and `"min"` use values of rows with any overlap; they do not weight
  extrema by overlap width.
- `"max_abs_delta"` uses the largest finite absolute aligned REF/ALT delta and
  is pair-only.
- No covered bases returns `nan`.

For indels, `TrackWindowScorer`, `TrackFeatureScorer`,
`TrackAllFeaturesScorer`, and `TrackEffectPeakScorer` use
`SequencePair.coordinate_mapper()` to map REF windows into ALT coordinates.
Aligned delta rows remain in REF coordinates. Deleted/uncovered intervals are
represented as non-finite rather than filled with zero.

## Expression scorers

### `ExpressionScorer`

Absolute single-prediction scorer:

```python
ExpressionScorer(
    output="expression",
    transform="none",
    name="expression",
)
```

| Argument | Purpose |
| --- | --- |
| `output` | Name passed to `Prediction.scalar()`. |
| `transform` | `"none"` keeps the value. `"log2"` computes `log2(max(value, 0) + 1)` for each scalar/list element. Other values raise when scoring. |
| `name` | `PredictionScoringResult.name`. |

`score_prediction(prediction)` returns `PredictionScoringResult` and records the
mode, output, and transform in provenance. It needs only a retained scalar
output (`requires_tokens = False`), which lets `SequenceOptimizer` request
scalar inference retention.

This class does not implement pair `score()`.

### `ExpressionDeltaScorer`

Pair effect:

```python
ExpressionDeltaScorer(
    output="expression",
    sign="alt-ref",
    transform="none",
    name="expression_delta",
)
```

`score(prediction)` obtains REF and ALT via `Prediction.scalar(output)`, applies
the transform to each allele before subtraction, aligns list outputs to the
shorter length, and returns `ScoringResult`.

Transforms:

| Value | Behavior |
| --- | --- |
| `"none"` | No transform. |
| `"log2"` | `log2(max(value, 0) + 1)`. |
| `"zscore"` | Declared in the type annotation but the implementation always raises because no fitted reference distribution is stored. |

## Token and fixed-window track scorers

### `TokenWindowScorer`

Scores raw model-token positions without using token-to-base coordinate mapping:

```python
TokenWindowScorer(
    output="atac",
    tokens_left=37,
    tokens_right=37,
    center_token="middle",
    aggregate="sum",
    sign="alt-ref",
    channel=0,
    name="token_window",
)
```

| Argument | Current behavior |
| --- | --- |
| `tokens_left`, `tokens_right` | Define `[common_middle - tokens_left, common_middle + tokens_right)`, clipped to the shorter allele token length. |
| `aggregate` | `"sum"`, `"mean"`, or `"max"` reduce each allele before subtraction; `"max_abs_delta"` computes the maximum absolute token-wise delta. |
| `channel` | Select/average logits as described in shared arguments. |
| `sign`, `name` | Pair sign and result name. |
| `output` | Stored in provenance only; `score()` directly reads `prediction.ref.logits` and `prediction.alt.logits`. |
| `center_token` | Public field but currently not consulted; `score()` always uses the common token-list midpoint. |

`score()` returns a token-unit `ScoreWindow`. Logits must be retained and must be
2-D or 3-D.

### `TrackWindowScorer`

Scores a coordinate-aware track inside one sequence window:

```python
TrackWindowScorer(
    track="atac",
    center="variant",
    width_bp=None,
    left_bp=None,
    right_bp=None,
    aggregate="sum",
    sign="alt-ref",
    channel=0,
    name="track_window",
)
```

Window arguments:

- Supply `width_bp` for a symmetric fixed-width interval.
- Or supply both `left_bp` and `right_bp` for
  `[center - left_bp, center + right_bp)`.
- Supplying `width_bp` together with either asymmetric side raises.
- If all three window fields are `None`, `width_bp` is set to `501`. Supplying
  only one of `left_bp`/`right_bp` also enters that defaulting branch and then
  raises because a width and one asymmetric side are simultaneously present.

Pair-center resolution:

- integer: use directly;
- `Feature`: use midpoint;
- `"variant"`: use `SequencePair.variant_feature()`;
- another string: strip `feature:` and find an exact name/type on REF.

Public methods:

| Method | Result | Used elsewhere |
| --- | --- | --- |
| `window_for(pair_prediction)` | Return REF-side `ScoreWindow`. | `TrackFeatureBuilder` uses it to set its binning span. |
| `window_for_prediction(prediction)` | Resolve a window on one prediction's retained sequence. |
| `score(pair_prediction)` | Build REF/ALT/delta rows, map the REF window to ALT, aggregate, and return `ScoringResult` with allele-specific windows. |
| `score_prediction(prediction)` | Return one absolute `PredictionScoringResult`; rejects `"max_abs_delta"`. |

Pair scoring requires retained `PairPrediction.pair`, tokens, and logits.
Absolute coordinate scoring requires a retained sequence, tokens, and logits.

## Adaptive and feature-based track scorers

### `TrackEffectPeakScorer`

Discovers strongest finite local maxima in an aligned delta track, builds windows
around selected peaks, merges overlapping windows, and compares aggregate REF
and ALT signal over the union.

```python
TrackEffectPeakScorer(
    track="atac",
    channel=0,
    search_center="variant",
    search_width_bp=10_001,
    window_bp=501,
    max_points=1,
    min_distance_bp=501,
    min_effect_fraction=0.8,
    selection="max_abs_delta",
    direction="same_as_primary",
    smoothing_bp=None,
    spatial_aggregate="sum",
    comparison="difference",
    combine_points="union",
    pseudocount=1.0,
    sign="alt-ref",
    name="track_effect_peaks",
)
```

| Argument | Purpose and valid values |
| --- | --- |
| `search_center` | Integer, `Feature`, `"variant"`, or exact REF feature name/type used to center peak search. |
| `search_width_bp` | Positive, sequence-clipped REF search width. |
| `window_bp` | Positive window width around each selected peak. |
| `max_points` | Positive maximum number of peaks retained. |
| `min_distance_bp` | Non-negative minimum center distance. A candidate closer than this to an already selected stronger peak is rejected. |
| `min_effect_fraction` | `(0, 1]`; candidate strength must be at least this fraction of the primary peak's strength. |
| `selection` | Only `"max_abs_delta"` is implemented. |
| `direction` | `"same_as_primary"` rejects nonzero opposite-sign peaks; `"any"` permits either direction. |
| `smoothing_bp` | `None` uses raw delta for selection. A positive integer uses an overlap-weighted mean delta around each candidate for ranking/sign filtering. Final window aggregation remains on native tracks. |
| `spatial_aggregate` | `"sum"` or `"mean"` across the union of selected windows. |
| `comparison` | `"difference"` computes ALT minus REF. `"log2_ratio"` computes `log2((ALT + pseudocount)/(REF + pseudocount))`. |
| `combine_points` | Only `"union"` is implemented; overlapping point windows are merged so bases are not counted twice. |
| `pseudocount` | Must be positive for `"log2_ratio"`; shifted REF and ALT totals must also be positive at scoring time. |
| `sign` | `"alt-ref"` or `"ref-alt"`. It also controls the aligned delta used for peak direction/ranking. |

`score()` returns:

- the combined scalar in `result.score`;
- a bounding legacy `score_window`;
- precise merged `ref_score_windows` and mapped `alt_score_windows`;
- one `result.features` entry per unmerged selected peak, including its
  position, candidate/selection effect, strength, per-point score, and windows;
- REF, ALT, and aligned delta rows;
- warnings for deleted/uncovered search intervals or no finite peak.

No finite peak returns `score=nan` and an empty feature list.

### `TrackFeatureScorer`

Aggregates one track over one or more selected annotations:

```python
TrackFeatureScorer(
    features,
    track="atac",
    aggregate="sum",
    sign="alt-ref",
    channel=0,
    name="track_feature",
)
```

| Argument | Purpose |
| --- | --- |
| `features` | One string/`Feature`, or a sequence. A string may start with `feature:` and otherwise matches exact feature name or type. All features matching a selector are used. |
| `aggregate` | `"sum"`, `"mean"`, `"max"`, `"min"`, `"max_abs_delta"`, `"auc"`, or `"weighted_sum"`. |
| Other fields | Shared track/channel/sign/name semantics. |

Overlapping/adjacent selected windows are merged before aggregation.
Any selector with no match raises.

| Method | Behavior |
| --- | --- |
| `score(pair_prediction)` | Resolve selectors on REF, map merged windows to ALT, calculate a pair effect, and store exact allele windows/tracks/provenance. |
| `score_prediction(prediction)` | Resolve selectors on the prediction's own retained sequence and return an absolute score; `"max_abs_delta"` is rejected. |

### `TrackAllFeaturesScorer`

Scores every feature independently:

```python
TrackAllFeaturesScorer(
    track="atac",
    aggregate="sum",
    sign="alt-ref",
    channel=0,
    feature_source="ref",
    name="track_all_features",
)
```

| Argument | Purpose |
| --- | --- |
| `feature_source` | `"ref"` enumerates REF features and maps each interval to ALT. `"alt"` enumerates ALT and maps to REF. Validated at construction. |
| `aggregate` | Same options as `TrackFeatureScorer`. |
| Other fields | Shared semantics. |

Duplicate feature names become stable keys such as `name#1`, `name#2`.
`score()` returns a `{feature_key: score}` mapping and detailed feature entries
with original feature data and both allele windows.

`score_prediction()` ignores the pair-only meanings of `feature_source` and
`sign`, uses the prediction's own features, and rejects `"max_abs_delta"`.

`ScoringResult.feature_names` and `ScoringResult.feature(name)` are designed for
the feature entries returned by this class and also work with peak entries.

## Regression features and scores

### `TrackFeatureBuilder`

Builds a fixed-length numeric feature vector by binning a token-index-aligned
delta track:

```python
TrackFeatureBuilder(
    track="atac",
    center="variant",
    width_bp=5000,
    bin_size=50,
    aggregate="mean",
    channel=0,
)
```

`build(pair_prediction)`:

1. uses `TrackWindowScorer.window_for()` to resolve the span;
2. calls `PairPrediction.delta_track()`—this is index-aligned, not the
   coordinate-mapped delta used by coordinate-aware scorers;
3. walks bins of `bin_size`;
4. selects rows with any overlap;
5. uses `"sum"` or otherwise the arithmetic mean of selected token deltas;
6. fills bins with no values using `0.0`.

It returns `(feature_values, metadata_rows)`, where each metadata row has
`feature_start`, `feature_end`, and `feature_center`. Supply positive
`width_bp`/`bin_size`; no explicit constructor validation is present.

### `RegressionScorer`

```python
RegressionScorer(
    model,
    feature_builder,
    name="regression_score",
)
```

| Argument | Purpose |
| --- | --- |
| `model` | Fitted object with `predict(X)`. |
| `feature_builder` | `TrackFeatureBuilder` used to create one row of inputs. |
| `name` | Result name. |

`load(path, *, feature_builder, name="regression_score")` loads with joblib. If
the loaded object is a mapping containing `"model"`, that entry is used;
otherwise the whole object is the model.

`score(pair_prediction)` converts features to a NumPy `(1, n_features)` array,
takes the first flattened prediction as a float, and returns `ScoringResult`
with bin metadata in `features`.

## `ScorerSet`

Runs several scorers against the same prediction:

```python
ScorerSet(scorers)
```

`scorers` may be:

- a mapping whose keys become report keys; or
- a sequence whose `scorer.name` values become keys.

Methods:

| Method | Behavior | Used elsewhere |
| --- | --- | --- |
| `score(pair_prediction)` | Return `VariantReport` from every `scorer.score()`. |
| `score_prediction(prediction)` | Require every scorer to implement `score_prediction()` and return non-nested `PredictionScoringResult`; return `PredictionReport`. |
| `score_retained(pair_prediction, retention_policy)` | Score/compact each pair result before running the next scorer, limiting simultaneous retained payloads. Nested `VariantReport` results are rejected. | `VariantInterpreter` detects and uses this in non-full retention mode. |

ISM recognizes `ScorerSet`, creates one dataframe column per configured key/name,
and requires all resulting scores to be scalar.

## `VariantInterpreter`

High-level scoring orchestrator:

```python
VariantInterpreter(model: SequenceModel)
```

The constructor stores the model used by all workflows.

### Common scoring arguments

| Argument | Purpose |
| --- | --- |
| `condition` / `conditions` | Singular broadcast condition or plural row-wise conditions. Pass exactly one form. Accepted items are `Condition`, string, or mapping. |
| `scorer` | Object with pair `score()`; often one of the classes above or `ScorerSet`. |
| `center` | Tokenization center forwarded to `SequenceModel.predict_multiple_pairs()`. Default `"tss"` can fall back to `"variant"` only under the model's exact pair fallback rule. |
| `grouping`, `pair_execution` | Inference choices documented in the inference README. |
| `preprocessing_workers`, `preprocessing_backend`, `prefetch_batches`, `show_progress` | Batch preprocessing/progress controls forwarded to inference. |
| `max_records_per_forward`, `max_pairs_per_forward` | Forward limits selected by pair execution. |
| `retention` | Preset/policy. Scorers always receive full predictions; outputs are compacted afterward. |
| `on_error` | Batch methods accept `"raise"`, `"warn"`, `"skip"`, or `"exit"`. `"warn"`/`"skip"` process one item at a time; warn emits `RuntimeWarning`, skip is silent. `"exit"` raises `SystemExit`. |

### `score_variant(...)`

```python
score_variant(
    variant,
    *,
    context=None,
    condition,
    scorer,
    genome=None,
    center="tss",
    grouping="no_grouping",
    pair_execution="separate",
    coordinate_system="auto",
    retention=None,
    **sequence_pair_kwargs,
)
```

Normalize one variant record, call `Variant.to_sequence_pair()`, predict, score,
prefix the result name with a stable input label, compact, and return one
`ScoringResult` or `VariantReport`.

`sequence_pair_kwargs` are forwarded to `Variant.to_sequence_pair()` and may
select its `window_bp` or primer-defined modes.

### `score_variants(...)`

Adds plural `conditions`, batch/preprocessing limits, progress, and `on_error`;
returns a list in the order of successfully scored inputs.

Accepted variant record shapes:

- `Variant`;
- four-field variant string accepted by `Variant.from_str()`;
- mapping/object with `variant`;
- mapping/object with `chrom`, `pos`, `ref`, and `alt`.

Coordinate behavior:

- strings use `Variant.from_str()`, where `"auto"` and `"1-based"` subtract
  one;
- mapping/object field rows subtract one only for `"1-based"`;
- existing `Variant` objects are used unchanged.

For `on_error="warn"` or `"skip"`, failed rows are absent from the returned
list.

### `score_sequence_pair(...)`

```python
score_sequence_pair(
    pair,
    *,
    condition,
    scorer,
    center="tss",
    grouping="no_grouping",
    pair_execution="separate",
    label=None,
    retention=None,
)
```

Scores one prebuilt pair. `label` overrides automatic result-name labeling.
The one-item path disables progress and prefetch.

### `score_sequence_pairs(...)`

Plural prebuilt-pair method with common batch options and `on_error`.
Automatic labels prefer `pair.variant.id`, then pair metadata `name`/`id`/
`variant_id`, then an equal REF/ALT sequence name, and finally `pair_<index>`.

## Result classes

### `ScoreWindow`

Frozen value:

```python
ScoreWindow(start, end, center=None, units="sequence")
```

It stores a scorer-used interval. `units` is typically `"sequence"` or
`"token"`. Construction does not validate bounds. `to_dict()` serializes all
fields.

### `DisplayWindow`

Frozen display-only value:

```python
DisplayWindow(start, end, center=None, annotation_which="ref")
```

Requires `start >= 0`, `end > start`, and `annotation_which` equal to `"ref"` or
`"alt"`. It changes plot limits only; it never changes the score.

### `PredictionScoringResult`

Absolute-score container:

```python
PredictionScoringResult(
    name,
    score,
    prediction,
    score_window=None,
    score_windows=(),
    track=None,
    features=None,
    warnings=(),
    provenance=None,
)
```

| Method/property | Result |
| --- | --- |
| `condition_name` | `prediction.condition.name`. |
| `sequence_name` | Retained sequence name or `None`. |
| `to_frame()` | One-row pandas DataFrame with name, score, condition, and sequence. |
| `to_json(path=None)` | JSON string; optionally writes it. Includes windows, identities, warnings, and provenance, not full prediction/track/features. |

### `ResultIdentity`

Frozen lightweight variant/condition identity used after full prediction
retention is removed:

```python
ResultIdentity(
    condition,
    variant_id=None,
    chrom=None,
    pos=None,
    ref=None,
    alt=None,
)
```

`id` aliases `variant_id`; `from_prediction()` extracts identity from a retained
pair; `to_dict()` serializes variant fields.

### `ScoringResult`

Pair-score container:

```python
ScoringResult(
    name,
    score,
    prediction,
    score_window=None,
    ref_track=None,
    alt_track=None,
    delta_track=None,
    features=None,
    warnings=(),
    provenance=None,
    display_window=None,
    identity=None,
    ref_score_windows=(),
    alt_score_windows=(),
)
```

Key properties:

| Property | Behavior |
| --- | --- |
| `variant` | Return `prediction.pair.variant`, or retained `ResultIdentity`; otherwise raise `RetainedDataError`. |
| `condition` | Return full `Condition`; requires retained prediction. |
| `condition_name` | Return name from prediction or lightweight identity. |
| `input_ref`, `input_alt` | Return annotated allele inputs; require retained prediction and pair. |
| `ref_prediction`, `alt_prediction` | Return allele predictions; require retained prediction. |
| `feature_names` | Keys from feature entries containing both `key` and `score`. |

Selection/view methods:

| Method | Arguments and behavior |
| --- | --- |
| `feature(name)` | Return a scalar `ScoringResult` view for one feature/peak entry, preserving payloads and selecting its allele windows. |
| `window(start=..., end=..., annotation_which="ref")` | Create a display view in REF or ALT local coordinates. Both bounds are required and must lie within the selected retained sequence. |
| `window(center=..., width_bp=..., annotation_which="ref")` | Resolve an integer or feature/feature-result key, clip/shift a positive-width view to sequence bounds, and attach `DisplayWindow`. |

Export methods:

| Method | Result |
| --- | --- |
| `to_frame()` | One-row pandas DataFrame with score, condition, and variant fields. |
| `to_json(path=None)` | JSON string and optional file including score/display windows, identity, warnings, and provenance. Tracks/features/full prediction are not serialized. |

Plot methods:

| Method | Important arguments and requirements |
| --- | --- |
| `plot_sequence(which=None, save_path=None, **kwargs)` | Delegate to `AnnotatedSequence.plot_annotations()`. Default allele comes from display window or REF. Requires retained pair. |
| `plot_variant(flank=40, save_path=None, **kwargs)` | Delegate to `SequencePair.plot_difference()`. Requires retained pair. |
| `plot_tracks(figsize=(12, 6.4), title=None, save_path=None)` | Plot REF, ALT, and delta rows; requires all three retained tracks. |
| `plot_delta_track(figsize=(12, 3), title=None, save_path=None)` | Still calls the same all-track requirement, so all three tracks must be retained even though only delta is drawn. |
| `plot_summary(figsize=(12, 8), include_annotations=False, annotation_which=None, annotation_kwargs=None, save_path=None)` | Plot variant plus available tracks, optionally annotations. Without tracks it falls back to variant-only or variant-plus-annotations. |
| `save_plots(directory)` | Create the directory, save a variant plot, and save track/delta plots only when all three track fields are non-`None`; return written paths. |

Precise `ref_score_windows`/`alt_score_windows` are used for track shading.
`score_window` remains the broad/legacy field.

### `VariantReport`

```python
VariantReport(results: Mapping[str, ScoringResult])
```

| Method | Behavior |
| --- | --- |
| `scores()` | Return `{report_key: result.score}`. |
| `to_frame()` | Concatenate result frames with pandas. |
| `report[name]` | Return one result. |
| `plot_summary(save_path=None, **kwargs)` | Horizontal bar chart. List scores use their first item; mapping scores are not converted and therefore are not suitable for this plot. |

### `PredictionReport`

Same mapping pattern for `PredictionScoringResult`, with `scores()`,
`to_frame()`, and indexed access. It has no report-level plot method.

## Scoring retention helpers

Exported from `gena_expression.scoring`:

| Function | Behavior |
| --- | --- |
| `retain_scoring_result(result, retention)` | Apply a `RetentionPolicy` or `ScoringRetention`, optionally compact prediction, windows, tracks, feature detail, warnings, and provenance while preserving/creating `ResultIdentity`. |
| `retain_scoring_output(output, retention)` | Apply the helper to one `ScoringResult` or every result in `VariantReport`. |

These functions return dataclass copies. `VariantInterpreter` invokes them after
scoring; scores are computed from full inputs.

## Internal implementation types

`Scorer` is the protocol used for typing but is not exported through
`scoring.__init__`. `_TrackIntervalIndex` is an internal weighted interval
index. It uses prefix integrals for sorted, non-overlapping, finite rows and
falls back to scan semantics otherwise.

## Example

```python
from gena_expression import (
    ExpressionDeltaScorer,
    ScorerSet,
    TrackWindowScorer,
    VariantInterpreter,
)

scorers = ScorerSet(
    {
        "expression": ExpressionDeltaScorer(),
        "accessibility": TrackWindowScorer(
            center="variant",
            width_bp=501,
            aggregate="sum",
        ),
    }
)

report = VariantInterpreter(model).score_sequence_pair(
    pair,
    condition=condition,
    scorer=scorers,
    center="variant",
)

report.scores()
```

`model`, `pair`, and `condition` are existing `SequenceModel`, `SequencePair`,
and condition-compatible objects.

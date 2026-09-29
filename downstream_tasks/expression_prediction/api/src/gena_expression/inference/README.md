# Model inference

The `gena_expression.inference` subpackage loads a sequence model, converts DNA
and condition descriptions to model inputs, plans forward batches, and returns
prediction containers.

The package-level [README](../README.md) documents `Condition`,
`AnnotatedSequence`, `SequencePair`, tracks, and retention configuration. This
README focuses on inference-specific classes and arguments.

## Module map

| Module | Contents |
| --- | --- |
| `model.py` | `SequenceModel` loading, preprocessing, batching orchestration, and public prediction methods |
| `tokenization.py` | `CenteredTokenizer` and `TokenizedSequence` |
| `outputs.py` | `Prediction`, `ExpressionPrediction`, `PairPrediction`, and prediction-retention helpers |
| `batching.py` | Grouping/pair/preprocessing type aliases, forward-batch records, progress, and worker functions |
| `__init__.py` | Public inference exports |

## Shared option sets

### Grouping modes

Prediction methods accept:

| `grouping` | Forward-planning behavior |
| --- | --- |
| `"serial"` | One independent sequence/condition row per forward. In joint pair mode, one complete REF/ALT pair per forward. |
| `"no_grouping"` | Batch independent rows without a repeated-sequence or repeated-condition shortcut. `max_records_per_forward` bounds rows in ordinary/separate execution. |
| `"condition"` | Group identical rendered condition text, encode that description once, and reuse it for the group's DNA rows. |
| `"sequence"` | Group identical token-ID sequences, encode the DNA once, and reuse it for multiple descriptions. Not supported with joint pair execution. |
| `"auto"` | Ordinary rows choose the grouping that saves more repeated encodings, preferring condition grouping on a tie. Joint pairs choose condition grouping only when a condition repeats across pairs; otherwise they use pair batching. |

Grouping uses the exact condition text rendered by the model's snapshotted
formatter for condition identity and token IDs for sequence identity. It is not
based only on `Condition.name` or `AnnotatedSequence.name`.

### Pair execution

| `pair_execution` | Behavior | Batch limit |
| --- | --- | --- |
| `"separate"` | Predict all reference rows, then all alternative rows. This is the default two-pass protocol. | `max_records_per_forward` |
| `"joint"` | Interleave each REF immediately before its ALT and keep complete pairs inside one physical forward. | `max_pairs_per_forward`; physical DNA rows are twice this number |

Joint execution rejects `grouping="sequence"` and rejects
`max_records_per_forward`. Separate execution rejects `max_pairs_per_forward`.

### CPU preprocessing

| Argument | Meaning |
| --- | --- |
| `preprocessing_workers: int` | Number of CPU workers. `0` runs preprocessing synchronously. Values must be non-negative. |
| `preprocessing_backend` | `"process"` creates spawned process workers that reload tokenizers from their paths; `"thread"` uses thread workers sharing the configured centered tokenizer. |
| `prefetch_batches: int` | Number of future CPU batches prepared while the current batch runs. `0` disables look-ahead; values must be non-negative. |
| `show_progress: bool` | Enable `tqdm` progress for multi-item stages. |

Process preprocessing requires reloadable tokenizer `name_or_path` values, or
equivalent paths stored in `SequenceModel.provenance`.

### Retention

Each public prediction method accepts `retention=None`, `"full"`,
`"scalars"`, a `RetentionMode`, or a `RetentionPolicy`.

- `None` resolves the current context-local package policy.
- `return_tokens=False` suppresses token retention even when the policy would
  otherwise keep tokens.
- Track extraction requires both retained logits and `TokenizedSequence`.
- Pair predictions retain their original `SequencePair` only when sequence
  retention is enabled.

## Tokenization

### `CenteredTokenizer`

Converts one sequence into a bounded token window around a biological center.

Constructor:

```python
CenteredTokenizer(
    dna_tokenizer,
    dna_max_seq_len,
    token_len_for_fetch,
    num_before,
    cls_id=None,
    sep_id=None,
    pad_id=None,
)
```

| Argument | Purpose |
| --- | --- |
| `dna_tokenizer` | Tokenizer exposing `encode_plus()`, `decode()`, and special-token IDs. |
| `dna_max_seq_len` | Total model DNA token budget including CLS and SEP. The DNA-token budget is `dna_max_seq_len - 2`. |
| `token_len_for_fetch` | Number of bases fetched per requested upstream token. It controls the initial upstream DNA substring, not final token widths. |
| `num_before` | Maximum upstream DNA tokens retained before the center. The remainder of the DNA-token budget is filled downstream. |
| `cls_id`, `sep_id`, `pad_id` | Explicit special IDs. `None` uses the corresponding tokenizer attribute. `pad_id` is stored for integration but `tokenize()` itself emits no padding. |

Public methods:

#### `tokenize(sequence, *, center="tss", strand="+", return_offsets=True) -> TokenizedSequence`

| Argument | Purpose |
| --- | --- |
| `sequence` | `AnnotatedSequence` or a plain string. Plain strings are wrapped without features, so a named center such as `"tss"` cannot resolve unless the sequence is annotated. |
| `center` | Integer local coordinate; `Feature` midpoint; `"middle"`/`"midpoint"`; or a named feature resolved by `AnnotatedSequence.resolve_center()`. |
| `strand` | Exact `"-"` switches to the reverse-oriented tokenization branch. Use `"+"` or `"-"`. |
| `return_offsets` | Forwarded as `return_offsets_mapping` to both tokenizer calls. The fallback when offsets are absent uses one-base placeholder spans. |

The method:

1. fetches/tokenizes the upstream side and keeps the last `num_before` DNA
   tokens;
2. tokenizes the downstream side and keeps enough tokens to fill the remaining
   DNA budget;
3. reverses record order for minus-strand processing;
4. assigns sequence-local `start`, `end`, `center`, `input_position`,
   `feature_names`, and source name to each token row;
5. adds CLS and SEP IDs outside the DNA token records.

It raises when no DNA tokens are produced. It does not add model batch padding;
`SequenceModel` performs padding within a forward batch.

#### `reverse_complement(sequence)`

Static helper for `ACGTN` uppercase DNA. It is used internally by minus-strand
tokenization.

### `TokenizedSequence`

Mutable dataclass returned by `CenteredTokenizer`:

```python
TokenizedSequence(
    input_ids,
    attention_mask,
    tokens,
    source,
    center,
    strand="+",
)
```

| Field | Meaning |
| --- | --- |
| `input_ids` | One-dimensional PyTorch long tensor including CLS/SEP. |
| `attention_mask` | Same-length tensor of ones before batch padding. |
| `tokens` | DNA-token coordinate records; excludes CLS/SEP. |
| `source` | Original/wrapped `AnnotatedSequence`. |
| `center` | Resolved local sequence coordinate. |
| `strand` | Requested tokenization orientation. |

Methods:

| Method | Result | Used elsewhere |
| --- | --- | --- |
| `as_model_inputs()` | `{"dna_input_ids": input_ids, "dna_attention_mask": attention_mask}`. |
| `token_at_sequence_position(position)` | First model `input_position` whose token covers the base, or `None`. |
| `to_frame()` | pandas DataFrame of token records. |

`SequenceModel` stores this object on predictions when token retention is
enabled. `Prediction.track()` uses its `input_position` values to select model
logits.

## `SequenceModel`

### Construction with runtime objects

```python
SequenceModel(
    model,
    dna_tokenizer,
    description_tokenizer,
    *,
    dna_max_seq_len,
    desc_max_seq_len,
    token_len_for_fetch,
    num_before,
    device=None,
    output_names=None,
    provenance=None,
    description_formatter=None,
)
```

| Argument | Purpose |
| --- | --- |
| `model` | PyTorch-like model with `.to()`, `.eval()`, and the expected callable signature. Construction moves it to `device` and switches to evaluation mode. |
| `dna_tokenizer` | DNA tokenizer used to construct `CenteredTokenizer`. |
| `description_tokenizer` | Condition-description tokenizer. |
| `dna_max_seq_len` | Total DNA token budget, including two special tokens. |
| `desc_max_seq_len` | Truncation length for description tokenization. |
| `token_len_for_fetch`, `num_before` | Passed to centered DNA tokenization. |
| `device` | Explicit PyTorch device string. `None` selects `"cuda"` when available, otherwise `"cpu"`. |
| `output_names` | Copied output-name mapping. Default is `{"expression": 0, "track": 0, "atac": 0}`. Current expression/track access uses fixed prediction conventions rather than indexing through this mapping. |
| `provenance` | Base metadata copied into retained prediction provenance and used as a tokenizer-path fallback for process workers. |
| `description_formatter` | Callable or `"/path/file.py::ClassName::static_method"`. `None` uses the formatter explicitly configured by `Condition.set_description_formatter()`; direct construction raises if none exists because it has no dataset config to inspect. |

### `SequenceModel.load(...)`

```python
SequenceModel.load(
    model_cls,
    checkpoint,
    config,
    dna_tokenizer,
    description_tokenizer,
    *,
    dna_max_seq_len,
    desc_max_seq_len,
    token_len_for_fetch,
    num_before,
    device=None,
    output_names=None,
    description_formatter=None,
)
```

| Argument | Purpose |
| --- | --- |
| `model_cls` | `"path/to/model.py::ClassName"` or `"package.module::ClassName"`. The separator `::` is required. |
| `checkpoint` | `.tensors` loads with `safetensors.torch.load_file`; any other suffix loads through `torch.load(..., weights_only=True)`. |
| `config` | Hydra config file. The code composes its file name from its parent directory and instantiates `experiment_config["model_kwargs"]`. |
| `dna_tokenizer`, `description_tokenizer` | Paths/names passed to `AutoTokenizer.from_pretrained()`. The description tokenizer is loaded with left padding. |
| `description_formatter` | Per-model formatter specification. `None` uses the current `Condition` runtime formatter, then falls back to `make_description_from_json` on the dataset class selected by `config`. |
| Remaining keyword arguments | Same runtime/tokenization meanings as the direct constructor. |

The method constructs the model class with the instantiated `model_kwargs`,
loads its state dictionary, builds tokenizers, records paths/class information
as provenance, and delegates to the constructor.

Description formatters are resolved in this order: the explicit
`description_formatter`, the current `Condition` runtime formatter, then the
configured dataset class. The final fallback reads the first
`train_dataset_*` entry, or the first `valid_dataset_*` entry when no training
entry exists, and imports its `*target*` through `hydra.utils.get_class()`
without constructing the dataset. During that import,
`${GENALM_HOME}/GENA_LM` is temporarily prepended to `sys.path`. A missing
environment variable, directory, dataset config, target, or callable
`make_description_from_json` raises before checkpoint loading. The selected
formatter is retained only by the new model and does not mutate `Condition`
runtime state.

Dependencies used by `load()` are PyTorch, Hydra, Safetensors, and Transformers.

### Description/tokenization methods

| Method | Arguments and result | Used elsewhere |
| --- | --- | --- |
| `make_description(condition)` | Return strings unchanged; render structured descriptions with the model's snapshotted formatter. |
| `make_description_from_json(meta)` | Render one metadata mapping with the model's formatter. |
| `tokenize_description(condition)` | Tokenize without padding, with truncation to `desc_max_seq_len`, returning description text plus `desc_input_ids` and `desc_attention_mask`. | Batch execution and description caching. |
| `tokenize_sequence(sequence, center="tss", strand="+")` | Delegate to the configured `CenteredTokenizer`. | `predict_sequence()` and preprocessing paths. |
| `clear_cuda_cache()` | Static helper calling `torch.cuda.empty_cache()` only when CUDA is available. The batch prediction path calls it in `finally`. |

### `predict_sequence(...)`

```python
predict_sequence(
    sequence,
    *,
    condition,
    center="tss",
    strand="+",
    grouping="no_grouping",
    return_tokens=True,
    retention=None,
) -> ExpressionPrediction
```

This is the one-row expression API. `condition` may be a `Condition`, string, or
metadata mapping. Strings/mappings are wrapped as a condition named
`"condition"`. `center`, `strand`, grouping, token return, and retention follow
the shared rules above.

### `predict_multiple_sequences(...)`

```python
predict_multiple_sequences(
    sequences,
    conditions=None,
    *,
    condition=None,
    center="tss",
    strand="+",
    grouping="no_grouping",
    preprocessing_workers=0,
    preprocessing_backend="process",
    max_records_per_forward=None,
    prefetch_batches=1,
    show_progress=True,
    return_tokens=True,
    retention=None,
) -> list[ExpressionPrediction]
```

Row construction accepts:

- many sequences plus one broadcast `condition=...`;
- one sequence plus many `conditions=[...]`;
- equal-length sequence and condition collections;
- one condition in `conditions=[...]`, which is broadcast.

Pass exactly one of `conditions` and `condition`. Incompatible lengths raise.
An empty sequence collection returns `[]` and requires the plural condition
input to be empty/absent.

`max_records_per_forward=None` leaves each resolved group unbounded. A positive
integer chunks each group while preserving returned input order.

### `predict_pair(...)`

```python
predict_pair(
    pair,
    *,
    condition,
    center="tss",
    grouping="no_grouping",
    pair_execution="separate",
    retention=None,
) -> PairPrediction
```

Thin one-item wrapper around `predict_multiple_pairs()`. It disables prefetch
and progress. The model resolves `center` on both alleles. If the requested
center is exactly `"tss"`, neither allele has it, and both alleles have a
`"variant"` feature, the method falls back to `"variant"`. Other missing centers
raise.

### `predict_multiple_pairs(...)`

```python
predict_multiple_pairs(
    pairs,
    conditions=None,
    *,
    condition=None,
    center="tss",
    grouping="no_grouping",
    pair_execution="separate",
    preprocessing_workers=0,
    preprocessing_backend="process",
    max_records_per_forward=None,
    max_pairs_per_forward=None,
    prefetch_batches=1,
    show_progress=True,
    retention=None,
) -> list[PairPrediction]
```

`condition`/`conditions` broadcast rules match the multi-sequence method.
Pair-execution, grouping, preprocessing, forward limits, progress, and retention
follow the shared option sections.

Separate execution performs two `_predict_sequence_tasks()` stages—references,
then alternatives—and reuses a description cache. Joint execution performs one
interleaved stage and prevents pair boundaries from crossing a forward batch.
Returned pair order always follows input order.

`SequenceModel.predict_multiple_pairs()` is the prediction engine used by
`VariantInterpreter`.

## Prediction containers

### `Prediction`

```python
Prediction(
    sequence,
    condition,
    logits,
    outputs,
    tokens=None,
    provenance=None,
    description_tokens=None,
)
```

| Field | Meaning |
| --- | --- |
| `sequence` | Retained `AnnotatedSequence`, or `None`. |
| `condition` | Normalized `Condition`; always present. |
| `logits` | CPU model logits after prediction, or `None` under retention. |
| `outputs` | Named output mapping. `SequenceModel` always places expression here and may include logits. |
| `tokens` | `TokenizedSequence`, required for track extraction. |
| `provenance` | Model and forward-batch metadata, when retained. |
| `description_tokens` | Encoded description payload, hidden from repr/comparison and optionally retained. |

Methods:

#### `scalar(name="expression")`

Return a Python float or list. The method first uses `outputs[name]`. For
`name="expression"` only, it can fall back to the first logit token when logits
remain. Unknown names raise `KeyError`; missing retained logits raise
`RetainedDataError`.

#### `track(name="track", channel=0) -> TrackPrediction`

Select logits at token `input_position`s:

- 3-D logits with one channel use that channel;
- 3-D multi-channel logits use their mean when `channel=None`, otherwise the
  selected channel;
- 2-D logits are indexed directly.

Token metadata and logits must both be retained. The `name` labels the returned
track; it does not select a separate output tensor.

Coordinate-aware scorers call this method for REF and ALT tracks.

#### `to_dict()`

Serialize the optional sequence, condition, simple numeric outputs, and
provenance. It does not serialize raw `tokens`, `description_tokens`, or a
separate logits field.

### `ExpressionPrediction`

Subclass of `Prediction` with:

```python
expression: float | list[float] = 0.0
```

`SequenceModel` derives it from the first model logit token.
`to_frame()` returns a one-row pandas DataFrame with condition name, expression,
and optional sequence name.

### `PairPrediction`

```python
PairPrediction(ref, alt, pair, condition)
```

| Field | Meaning |
| --- | --- |
| `ref`, `alt` | Allele predictions. |
| `pair` | Original `SequencePair`, or `None` when sequence retention is disabled. |
| `condition` | Shared `Condition`. |

Methods:

| Method | Arguments and behavior | Used elsewhere |
| --- | --- | --- |
| `delta_scalar(name="expression", sign="alt-ref")` | Subtract aligned scalar/list elements up to the shorter length. `"ref-alt"` negates the result. | Simple expression-effect consumers; `ExpressionDeltaScorer` computes its own transformed version. |
| `delta_track(name="delta_track", sign="alt-ref", channel=0)` | Build REF/ALT tracks, subtract values by list index up to the shorter length, and attach ALT token rows. | `TrackFeatureBuilder`; coordinate-aware scorers instead map intervals explicitly. |

### Prediction retention helpers

Exported from `gena_expression.inference`:

| Function | Behavior | Used elsewhere |
| --- | --- | --- |
| `retain_prediction(prediction, retention)` | Return a dataclass copy with fields converted/dropped according to `PredictionRetention`. |
| `retain_pair_prediction(prediction, retention)` | Apply the helper to both alleles and drop the source `SequencePair` when sequence retention is false. | Scoring-result compaction. |

## Internal batching classes and workers

`batching.py` defines the type aliases `GroupingMode`, `PairExecutionMode`, and
`PreprocessingBackend`; these are used in annotations but are not exported by
`inference.__init__`.

`_ForwardBatch` and names beginning with `_process_worker_` or `_thread_worker_`
are internal. Process workers reconstruct compact `AnnotatedSequence`/`Feature`
payloads and reload tokenizers; they never receive the CUDA model.

## Example

```python
from gena_expression import AnnotatedSequence, Condition, Feature, SequenceModel

sequence = AnnotatedSequence(
    "A" * 4096,
    name="example",
    features=(Feature("tss", 2048, 2049, type="tss"),),
)
condition = Condition("example", "cell type is example.")

prediction = model.predict_sequence(
    sequence,
    condition=condition,
    center="tss",
)

expression = prediction.expression
atac = prediction.track("atac", channel=0)
```

Here `model` must be an already constructed or loaded `SequenceModel`.

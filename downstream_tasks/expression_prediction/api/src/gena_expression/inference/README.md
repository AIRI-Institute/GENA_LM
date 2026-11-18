# Model inference

The `gena_expression.inference` subpackage loads a sequence model, converts DNA
and condition descriptions to model inputs, plans forward batches, and returns
prediction containers.

The package-level [README](../README.md) documents `Condition`,
`PooledCondition`, `AnnotatedSequence`, `SequencePair`, tracks, and retention
configuration. This
README focuses on inference-specific classes and arguments.

## Module map

| Module | Contents |
| --- | --- |
| `model.py` | `SequenceModel` loading, preprocessing, batching orchestration, and public prediction methods |
| `tokenization.py` | `CenteredTokenizer` and `TokenizedSequence` |
| `outputs.py` | `Prediction`, `ExpressionPrediction`, `PairPrediction`, and prediction-retention helpers |
| `batching.py` | Grouping, pair execution, CPU preprocessing, forward-batch records, and progress |
| `pooling.py` | Expansion and reduction of pooled description rows |
| `workers.py` | CUDA replica workers and bounded batch dispatch |
| `AG_model.py` | Module-level AlphaGenome adapter |
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

### Pooled conditions

Use a [`PooledCondition`](../README.md#pooledcondition) in any `SequenceModel`
prediction method. Each description is rendered and evaluated separately
through the usual batching and grouping path. For each logical prediction,
the model averages logits in FP32 across all members, before expression or
track values are extracted. Each member receives equal weight, including
repeated descriptions. The average excludes batch padding.

A pool counts as one entry for broadcast and output ordering. Forward limits
apply after expansion. Each member consumes one ordinary row or one whole
REF/ALT pair in joint execution. The model reduces REF and ALT
independently and attaches the original pool to both predictions and the
`PairPrediction`.

With description-token retention enabled, pooled predictions carry a
`{"members": [...]}` payload in the supplied description order. Retained
provenance contains per-member entries under `description_pool["members"]`.
With multiple GPUs, each member entry records its `forward_device`.
`make_description()` and `tokenize_description()` accept single descriptions.

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
    precision="default",
)
```

| Argument | Purpose |
| --- | --- |
| `model_cls` | `"path/to/model.py::ClassName"` or `"package.module::ClassName"`. The separator `::` is required. |
| `checkpoint` | `.tensors` loads with `safetensors.torch.load_file`; any other suffix loads through `torch.load(..., weights_only=True)`. |
| `config` | Hydra config file. The code composes its file name from its parent directory and instantiates `experiment_config["model_kwargs"]`. |
| `dna_tokenizer`, `description_tokenizer` | Paths/names passed to `AutoTokenizer.from_pretrained()`. The description tokenizer is loaded with left padding. |
| `description_formatter` | Per-model formatter specification. `None` uses the current `Condition` runtime formatter, then falls back to `make_description_from_json` on the dataset class selected by `config`. |
| `device` | A single PyTorch device, `None` for automatic selection, or a comma-separated [CUDA device list](#multi-gpu-inference). |
| `precision` | `"default"` or `"float32"`. See [precision](#precision). |
| Remaining keyword arguments | Same runtime/tokenization meanings as the direct constructor. |

The method uses the instantiated `model_kwargs` and restores the checkpoint
state dictionary. It builds tokenizers and records model class and path
information in provenance. With a CUDA list, the workers construct and
restore their own model replicas.

Description formatters are resolved in this order: the explicit
`description_formatter`, the current `Condition` runtime formatter, then the
configured dataset class. The final fallback reads the first
`train_dataset_*` entry, or the first `valid_dataset_*` entry when no training
entry exists, and imports its `_target_` through `hydra.utils.get_class()`
without constructing the dataset. During that import,
`${GENALM_HOME}/GENA_LM` is temporarily prepended to `sys.path`. A missing
environment variable, directory, dataset config, target, or callable
`make_description_from_json` raises before checkpoint loading. The selected
formatter is retained only by the new model and does not mutate `Condition`
runtime state.

Dependencies used by `load()` are PyTorch, Hydra, Safetensors, and Transformers.

### Precision

`precision="default"` preserves the configured attention backend and enables
BF16 autocast on CUDA. `precision="float32"` requests
`attn_implementation="sdpa"` from the model constructor, promotes parameters and
floating-point buffers before loading the checkpoint, and runs inference with
autocast disabled and PyTorch's SDPA math kernel selected. The model constructor
must support the SDPA option. For `ExpressionCounts`, this mode requires the
ModernGENA DNA encoder.

The process-wide CUDA TF32 settings still apply. For multi-GPU inference, workers
copy the parent's float32 matmul precision and cuDNN TF32 settings at load time.
Measure tolerances on the checkpoint and hardware you plan to use.
The FP32 smoke test uses a tiny local model on CPU and CUDA. The CUDA list
parity test requires two available GPUs.

### Multi-GPU inference

Pass a CUDA device list such as `device="cuda:0,2,5"` to `SequenceModel.load()`.
Use the `cuda:` prefix and distinct, visible PyTorch device indices. With
`CUDA_VISIBLE_DEVICES`, indices refer to the devices as numbered by PyTorch
inside that process.

Each selected GPU runs one resident model replica in a spawned process. Each
GPU must hold a full model replica and its forward batch. Tokenization and
description formatting stay in the parent. Workers take complete forward
batches. Each completed forward frees a slot for another batch, including
when an earlier forward is still running. The pool holds up to two unfinished
batches per selected GPU across all workers. Predictions are assembled by
planned batch and row IDs and returned in input order. `prefetch_batches`
controls CPU preprocessing look-ahead separately.

Use the context manager or `model.close()` to release all replicas. If a
worker fails, load a fresh model before retrying. A model
loaded with a CUDA list exposes its devices in `model.provenance["devices"]`.
Retained forward provenance records the worker device under `forward_device`.
Access the model through prediction methods because each replica lives in its
worker. In standalone scripts, start the workers under the
`if __name__ == "__main__":` guard.

```python
from gena_expression import SequenceModel


def predict_on_two_gpus(model_options, sequences, condition):
    options = {**model_options, "device": "cuda:0,2"}
    with SequenceModel.load(**options) as model:
        return model.predict_multiple_sequences(
            sequences, condition=condition, max_records_per_forward=4,
        )
```

With a CUDA list and no explicit limit, ordinary or separate execution caps
forwards at the smaller of 8 rows and the row count divided by the device
count, rounded up. Joint execution applies the same rule with a cap of 4
complete pairs. These counts are computed after condition pools expand into
their member rows or pairs. Explicit limits and serial grouping still apply.

### Description/tokenization methods

| Method | Arguments and result | Used elsewhere |
| --- | --- | --- |
| `make_description(condition)` | Return strings unchanged; render structured descriptions with the model's snapshotted formatter. |
| `make_description_from_json(meta)` | Render one metadata mapping with the model's formatter. |
| `tokenize_description(condition)` | Tokenize without padding, with truncation to `desc_max_seq_len`, returning description text plus `desc_input_ids` and `desc_attention_mask`. | Batch execution and description caching. |
| `tokenize_sequence(sequence, center="tss", strand="+")` | Delegate to the configured `CenteredTokenizer`. | `predict_sequence()` and preprocessing paths. |
| `clear_cuda_cache()` | Static helper that clears the local CUDA cache when CUDA is available. In-process batch prediction calls it in `finally`. Multi-GPU worker lifetime is controlled by `close()`. |

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

This is the one-row expression API. `condition` may be a `Condition`,
`PooledCondition`, string, or metadata mapping. Strings and mappings are
wrapped as a condition named `"condition"`. `center`, `strand`, grouping, token return, and retention follow
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

On a single device, `max_records_per_forward=None` leaves each resolved group
unbounded. A CUDA device list uses the [default batch cap](#multi-gpu-inference).
A positive integer chunks each group while preserving returned input order.

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
| `condition` | Normalized `Condition` or `PooledCondition`. Always present. |
| `logits` | CPU model logits after prediction, or `None` under retention. |
| `outputs` | Named output mapping. `SequenceModel` always places expression here and may include logits. |
| `tokens` | `TokenizedSequence`, required for track extraction. |
| `provenance` | Model and forward-batch metadata, when retained. |
| `description_tokens` | Encoded description payload, hidden from repr/comparison and optionally retained. A pool stores an ordered list under `members`. |

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
| `condition` | Shared `Condition` or `PooledCondition`. |

Methods:

| Method | Arguments and behavior | Used elsewhere |
| --- | --- | --- |
| `delta_scalar(name="expression", sign="alt-ref")` | Subtract aligned scalar/list elements up to the shorter length. `"ref-alt"` negates the result. | Simple expression-effect consumers; `ExpressionDeltaScorer` computes its own transformed version. |
| `delta_track(name="delta_track", sign="alt-ref", channel=0, *, alignment="coordinate")` | Return REF-local deltas using the pair's coordinate mapping and overlap-weighted ALT values. `alignment="index"` reproduces legacy token-by-token subtraction with ALT token rows. | Coordinate-aware scorers share the alignment implementation. `TrackFeatureBuilder` defaults to index alignment for existing regressors. |

Coordinate alignment maps each REF interval to its ALT interval. The result
subtracts the REF score from the mean ALT score over overlapping bases.
`sign="ref-alt"` reverses the subtraction. Rows carry REF and ALT values
and an `alignment_status` of `mapped`, `deleted`, or `uncovered`. Deleted
and uncovered intervals produce `NaN`. Both alleles need retained logits and
tokens. When a retention policy drops the source sequences while keeping
those data, `SequenceModel` retains the coordinate mapping needed by the
delta calculation.

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
payloads and reload tokenizers. GPU replicas use spawned workers in
`workers.py` with their own resident model instances.

## AlphaGenome adapter

Import `AlphaGenomeSequenceModel` from `gena_expression.inference.AG_model`.
The adapter takes an initialized AlphaGenome client and a GENA DNA tokenizer
identifier or path. Choose an output type such as `"DNASE"`. It converts
base-resolution outputs to GENA token spans by averaging bases covered by
each token.

Conditions carry ontology CURIEs, such as `"UBERON:0000178"`, in the condition
name or description, or under a key such as `ontology_terms` in a description
mapping or the condition metadata. The adapter issues serial calls and uses
separate REF/ALT execution. Pooled text conditions and the `SequenceModel.load()`
CUDA and precision options are specific to `SequenceModel`.

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

## Inference stress test

Run `python -m gena_expression.stress` to compare execution settings on the
same inputs. The CLI samples the requested number of JSON descriptions from
one directory and uses [`PromoterSampler`](../dna/README.md#promotersampler)
to draw transcript-oriented sequences with a retained local TSS. Every run
scores the same sequence-description combinations. An explicit seed makes the
draws reproducible for unchanged inputs.

Create a `model.json` file with the keyword arguments for
[`SequenceModel.load()`](#sequencemodelload). Substitute paths and token
limits for your checkpoint. Use the configured dataset formatter or provide
`description_formatter` as a static-method reference in the JSON.

```json
{
  "model_cls": "../expression_model_final.py::ExpressionCounts",
  "checkpoint": "checkpoint.pt",
  "config": "model.yaml",
  "dna_tokenizer": "dna-tokenizer",
  "description_tokenizer": "description-tokenizer",
  "dna_max_seq_len": 512,
  "desc_max_seq_len": 256,
  "token_len_for_fetch": 6,
  "num_before": 128,
  "device": "cuda:0"
}
```

Run the command from the API directory with an indexed FASTA and its transcript annotation.

```bash
python -m gena_expression.stress \
    --model-options model.json \
    --descriptions descriptions/ \
    --genome reference.fa --annotation transcripts.gtf \
    --n-descriptions 4 --n-sequences 20 --sequence-length 501 \
    --seed 42 --progress --output stress-results/
```

The default compares ungrouped, condition-grouped, sequence-grouped, and
automatic grouping against a serial reference. Each description file must
contain a JSON string or object. The serial reference uses the base model
options with one row per forward.

- `--progress` prints the active run, batch size, and grouping to stderr. It
  enables the model's tqdm bars for forward batches within each API call and
  labels each chunk when a run spans multiple calls. The first run is
  `serial` with batch size 1.
- `--track-channel 0` includes per-token track comparisons in addition to
  expression. Compared track coordinates must match across runs.
- `--pair-check` switches the workload to deterministic REF/ALT pairs and
  compares separate and joint forwards by default. It reports REF and ALT
  predictions separately.
- `--tss-window` changes the total sampling window. It defaults to
  `--sequence-length`. `--max-attempts` caps sampling when draws miss the TSS.
- `--records-per-call` limits the number of rows per API request and must
  cover the largest configured `batch_size`.

To choose comparison settings, pass `--runs runs.json` with a mapping from
run names to options. `serial` is reserved. Each run can set `grouping`,
`batch_size`, `preprocessing_workers`, `preprocessing_backend`,
`prefetch_batches`, `reverse_order`, and `model_overrides`. Pair checks
also accept `pair_execution`. The serial reference still runs.

```json
{
  "regrouped": {"grouping": "condition", "batch_size": 8, "reverse_order": true},
  "fp32": {"batch_size": 4, "model_overrides": {"precision": "float32"}},
  "two_gpus": {"batch_size": 4, "model_overrides": {"device": "cuda:0,2"}}
}
```

Omit runs unsupported by your checkpoint or visible devices. The example FP32
run needs a compatible model constructor. The two-GPU run needs CUDA indices
0 and 2 to be visible.

The output directory receives three files.

- `manifest.json` records the sampled inputs, source metadata, sequence hashes,
  seed, and model/run options. A row ID is `sequence index × description count
  + description index`, using zero-based indices.
- `predictions.csv` contains predictions and errors by run, row, and metric.
  Relative error is `abs(prediction - mean) / max(abs(mean), epsilon)`, where
  `mean` uses finite predictions for that exact output across runs.
  `serial_absolute_error` compares the output directly with the serial reference.
  `--epsilon` sets the denominator floor.
- `summary.json` reports counts, non-finite values, quantiles, and the largest
  relative error per run and metric.

Use a fresh output directory for each run. The tests verify the CLI and
reporting on synthetic fixtures. Numerical drift on a real checkpoint depends
on the checkpoint, hardware, and execution settings.

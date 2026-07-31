# Context construction

The `gena_expression.dna` subpackage turns genomic intervals, variant
records, MPRA fragments, and plasmid records into annotation-preserving
`AnnotatedSequence` and `SequencePair` objects.

The package-level [README](../README.md) defines the shared coordinate and
immutability conventions. This README covers only the classes and functions in
`dna/`.

## Module map

| Module | Contents |
| --- | --- |
| `builders.py` | `Context`, `GenomeContext`, and `PlasmidContext` |
| `genome.py` | `Genome`, `GenomeRegion`, `GenomeInterval`, `SafeHarborSite`, and internal GTF/tabix backends |
| `plasmid.py` | `PlasmidRecord`, `PlasmidCollection`, plasmid value classes, primer-tail inference, MCS insertion, and circular window helpers |
| `__init__.py` | The subpackage export list |

## Context-builder contract

### `Context`

`Context` is a typing `Protocol`, not a concrete base class. A compatible object
implements:

```python
build(
    variant: Variant,
    *,
    genome: Any | None = None,
    name: str | None = None,
) -> SequencePair
```

`Variant.to_sequence_pair(context=...)` and `VariantInterpreter` use this
contract. Runtime inheritance from `Context` is not required.

## Genomic contexts

### `Genome`

`Genome` provides FASTA-backed sequence access, chromosome-label normalization,
and optional GTF/GFF annotations.

Constructor:

```python
Genome(
    fasta_path,
    annotation=None,
    build=None,
    chrom_style="auto",
    cache=True,
)
```

| Argument | Purpose |
| --- | --- |
| `fasta_path: str \| Path` | Indexed FASTA path opened through `pysam.FastaFile`. |
| `annotation: str \| Path \| Mapping \| None` | Optional `.gtf`, `.gff`, `.gff3`, or corresponding `.gz` path; alternatively, an internal-compatible chromosome-to-record mapping. Other objects are rejected. |
| `build: str \| None` | Genome-build label copied into coordinate-segment and interval-feature metadata. It does not select a FASTA automatically. |
| `chrom_style: "auto" \| "chr" \| "no_chr"` | Controls which `chr`-prefixed/unprefixed alternatives `normalize_chrom()` tries. The constructor stores the value without separate validation. |
| `cache: bool` | If true, retain a FASTA handle until `close()`. If false, `_handle()` opens a new handle for an operation. |

Annotation preparation is active behavior:

- A plain supported GTF/GFF is bgzip-compressed to a sibling `.gz` file when
  needed.
- A compressed file without a tabix index is indexed.
- If indexing indicates an unsorted file, the code uses Polars to create a
  sorted compressed sibling and indexes that file.
- These steps require write permission beside the annotation input.
- GTF/GFF rows are converted from 1-based inclusive to 0-based half-open
  intervals.

Public methods:

#### `fetch(chrom, start, end, *, strand="+", pad="N") -> str`

| Argument | Purpose |
| --- | --- |
| `chrom` | Label normalized against FASTA references, including mitochondrial alternatives. |
| `start`, `end` | 0-based half-open genomic interval. Out-of-bounds portions are filled rather than rejected. |
| `strand` | Exact `"-"` reverse-complements the fetched/padded string; other values follow the forward branch. Use `"+"` or `"-"`. |
| `pad` | String repeated for bases before position 0 or after chromosome end. The method is designed for a one-character fill such as `"N"`. |

#### `sequence(chrom, start, end, *, strand="+", include_features=True, name=None) -> AnnotatedSequence`

Fetches on the forward strand, creates a genome `CoordinateMap`, optionally adds
a full-window `genome_interval` feature plus overlapping annotation features,
and finally reverse-complements the annotated object when `strand == "-"`.

`include_features=False` suppresses both the full-window feature and GTF/GFF
features. `name` becomes the returned sequence name.

This method is used by `GenomeContext.build()`.

#### `sequence_around(chrom, center, size, *, strand="+", include_features=True, center_feature_name=None, name=None)`

Computes `start = center - size // 2`, calls `sequence()` for exactly `size`
bases, and optionally adds a one-base feature at local `size // 2`.
`GenomeContext.sequence_at_tss()` uses this method with
`center_feature_name="tss"`.

#### `features(chrom, start, end, *, types=None, strand=None) -> list[Feature]`

Returns annotation rows overlapping the requested genomic interval, clipped and
shifted into local sequence coordinates.

| Argument | Purpose |
| --- | --- |
| `types` | Optional case-insensitive feature-type inclusion filter. |
| `strand` | Optional exact annotation-strand filter. |

Feature names are chosen from `gene_name`, then `transcript_name`, then
`gene_id`, and finally a coordinate-derived fallback. Original genomic
coordinates, score, frame, and parsed attributes are stored in metadata.

#### Other methods

| Method | Behavior | Used elsewhere |
| --- | --- | --- |
| `validate_ref(chrom, pos, ref, coordinate_system="0-based")` | Fetch the declared REF allele and emit `RuntimeWarning` on mismatch. `"1-based"` subtracts one; the other annotated value is `"0-based"`. | `Variant.validate()` implements the same kind of check through `Genome.fetch()`. |
| `normalize_chrom(chrom)` | Resolve a FASTA contig or raise with the candidates tried. | All FASTA access. |
| `chrom_length(chrom)` | Return the normalized FASTA contig length. |
| `close()` | Close a cached FASTA handle and a lazy tabix annotation handle. |

`pysam` is required for FASTA and tabix operations. Polars is additionally
required only for the unsorted-annotation fallback.

### `GenomeContext`

Immutable `Context` implementation for a fixed-length local genomic pair:

```python
GenomeContext(
    length,
    center="variant",
    strand="+",
    include_features=True,
    pad="N",
)
```

| Argument | Purpose |
| --- | --- |
| `length: int` | Number of reference bases requested from `Genome.sequence()`. It is also stored in pair metadata. No constructor validation is performed, so supply a positive value. |
| `center` | `"variant"` uses `variant.pos`; an integer is an absolute genomic center; any other string looks up `variant.metadata[str(center)]` and falls back to `variant.pos`. |
| `strand` | Passed to `Genome.sequence()` to orient the returned annotated sequence. |
| `include_features` | Passed to `Genome.sequence()`. |
| `pad` | Public dataclass field, but the current `build()` and `sequence_at_tss()` implementations do not forward or read it. Genome boundary padding therefore uses `Genome.fetch()`'s default `"N"`. |

Methods:

| Method | Arguments and behavior | Used elsewhere |
| --- | --- | --- |
| `build(variant, *, genome=None, name=None)` | Require `genome`, `variant.chrom`, and `variant.pos`; fetch a centered REF window, add a local `"variant"` feature, call `Variant.apply_to()` for ALT, and return a validated `SequencePair`. | `Variant.to_sequence_pair(window_bp=...)` constructs and invokes it; `VariantInterpreter` reaches it through `Variant.to_sequence_pair()`. |
| `sequence_at_tss(genome, chrom, tss, *, strand="+", name=None)` | Return a `length`-base `Genome.sequence_around()` result with a one-base local `"tss"` feature. |

`build()` does not perform a separate FASTA REF validation before applying the
variant. `Variant.apply_to()` warns if the fetched local REF slice differs.

### `SafeHarborSite`

Frozen value object holding:

```python
SafeHarborSite(
    abbrev_name,
    chromosome,
    start,
    end,
    category="",
    genes="",
)
```

Coordinates are 1-based inclusive. `GenomeRegion.SAFE_HARBORS` stores the named
site catalog used by `GenomeRegion`.

### `GenomeInterval`

Frozen 1-based inclusive interval with `chromosome`, `start`, and `end`.

`insertion_index` is a property returning
`((start - 1) + end) // 2`, a 0-based midpoint used by `GenomeRegion`.

### `GenomeRegion`

Builds a callable that pads an insert with genomic flanks at a named safe-harbor
site or explicit interval.

Constructor:

```python
GenomeRegion(
    genome_path,
    region=None,
    *,
    chromosome=None,
    position=None,
    start=None,
    end=None,
)
```

| Argument | Purpose |
| --- | --- |
| `genome_path` | Indexed FASTA path opened by the returned context callable. |
| `region` | Case/punctuation-insensitive safe-harbor name from `SAFE_HARBORS`. |
| `chromosome`, `position` | Explicit 1-based single-position insertion site. |
| `chromosome`, `start`, `end` | Explicit 1-based inclusive interval whose midpoint becomes the insertion point. |

Named and manual coordinates are mutually exclusive. Construction may omit a
site; a later `genomic_context()` call must then provide one.

#### `genomic_context(..., target_length) -> Callable`

Accepts the same site-selection arguments and returns a callable:

```python
context_for_sequence(sequence: str | AnnotatedSequence) -> AnnotatedSequence
```

`target_length` must be positive and at least the insert length. The callable:

1. validates and uppercases a plain string (`ACGTN` only), or preserves an
   `AnnotatedSequence`;
2. splits missing length between left and right flanks;
3. fetches the flanks without out-of-bounds padding;
4. adds `genome_flank` and `inserted_sequence` features;
5. concatenates all three annotated parts and adds insertion metadata.

This callable is a sequence-to-sequence context function. It is not the
`Context.build(Variant)` protocol.

## Plasmid records and contexts

### `FeatureMatch`

Frozen description of the reporter feature selected from a GenBank record:

| Field | Meaning |
| --- | --- |
| `key` | Parsed label. |
| `start`, `end` | 1-based inclusive GenBank coordinates. |
| `strand` | Integer `1` or `-1`. |
| `note` | Joined qualifier values. |

### `PlasmidMetadata`

Frozen routing value with `name` and `ncbi_link`. `PlasmidCollection` accepts
any mapping value exposing those two attributes, so this class is the direct
built-in representation rather than a mandatory base type.

### Primer-tail type aliases

Available from `gena_expression.dna`:

```python
PrimerTailEntry = tuple[str, str] | Mapping[str, object]
PrimerTails = Mapping[str, PrimerTailEntry]
```

A tuple means `(forward_tail, reverse_primer_tail)`. A mapping must at least
provide `"forward_tail"` and `"reverse_tail"` and may carry primer-match detail.
The reverse-primer tail is reverse-complemented when building the right scar.

### `PlasmidRecord`

Represents cleaned plasmid DNA plus optional parsed GenBank annotations.

Constructor:

```python
PlasmidRecord(
    sequence,
    *,
    name=None,
    genbank_text=None,
    reporter_feature_name="luc2",
    metadata=None,
)
```

| Argument | Purpose |
| --- | --- |
| `sequence` | Whitespace-stripped, uppercased plasmid DNA. Only `ACGTN` is accepted. |
| `name` | Record name; defaults to `"plasmid"`. |
| `genbank_text` | Optional flatfile text. When present, FEATURES are parsed immediately. |
| `reporter_feature_name` | Case-insensitive substring used to select the longest matching reporter feature. |
| `metadata` | Copied metadata attached to `annotated_sequence()`. |

The parser supports feature locations by extracting numeric endpoints and
recognizing an outer `complement(...)`. It does not construct a full GenBank
location AST; complex locations are represented by their minimum/maximum
numeric span.

Construction methods:

| Method | Arguments and behavior |
| --- | --- |
| `from_files(genbank_path, fasta_path=None, reporter_feature_name="luc2")` | Read GenBank text; read DNA from FASTA when supplied, otherwise from the GenBank `ORIGIN` block; choose name from `LOCUS` or the file stem. |
| `from_ncbi(ncbi_url, data_storage, reporter_feature_name="luc2")` | Parse accession from the final URL path component, cache `<accession>.gb` and `.fa` under `data_storage/accession/`, then call `from_files()`. Existing files are reused. |

Public instance methods:

| Method | Behavior | Used elsewhere |
| --- | --- | --- |
| `reporter_tss_index()` | Return reporter start minus one on forward strand or reporter end minus one on reverse strand. Raise if no reporter feature is available. | `plasmid_context()` centers its output here. |
| `annotated_sequence()` | Convert parsed GenBank records to 0-based half-open `Feature`s and return an `AnnotatedSequence`. | `plasmid_context()` starts from it. |
| `plasmid_context(*, context_size=None, circular=True, allow_repeats=False, primer_tails)` | Return a callable that inserts an MPRA fragment and centers the result on the reporter TSS. | `PlasmidContext` and `PlasmidCollection`. |

`plasmid_context()` arguments:

| Argument | Purpose |
| --- | --- |
| `context_size` | `None` returns one edited-plasmid length. A positive integer requests that many bases around the reporter. |
| `circular` | Enable wraparound while extracting the reporter-centered window. |
| `allow_repeats` | When the requested window exceeds the edited plasmid, allow repeated edited-plasmid copies if true. If false, outlying bases are filled from the original plasmid instead. |
| `primer_tails` | Element-keyed tail table used to locate/replace the MCS scars. |

The returned callable accepts `(mpra_fragment, element, *args, **kwargs)`.
`mpra_fragment` may already be annotated. Element lookup is normalized for case
and punctuation and supports a unique prefix match. The result includes a
one-base `reporter_tss` feature and `plasmid_context_debug` metadata.

### `PlasmidCollection`

Routes element names to lazily downloaded/cached plasmids.

Constructor:

```python
PlasmidCollection(
    data_storage,
    *,
    context_size=None,
    circular=True,
    allow_repeats=False,
    primer_tails,
    reporter_feature_name="luc2",
    element_to_plasmid,
)
```

| Argument | Purpose |
| --- | --- |
| `data_storage` | NCBI cache root passed to `PlasmidRecord.from_ncbi()`. |
| `context_size`, `circular`, `allow_repeats` | Forwarded to each cached `PlasmidRecord.plasmid_context()`. |
| `primer_tails` | Shared normalized element-to-tail mapping. |
| `reporter_feature_name` | Passed while loading each plasmid. |
| `element_to_plasmid` | Mapping from element label to an object with `name` and `ncbi_link` attributes. |

#### `element_name_based_context(mpra_fragment, element_name, *args, **kwargs)`

Normalize/resolve `element_name`, lazily load its source plasmid, reuse a cached
context callable by plasmid name, and return its annotated context. Extra
arguments are forwarded to that context callable.

This method is designed to be passed as a legacy sequence context function.

### `PlasmidContext`

Immutable `Context` implementation:

```python
PlasmidContext(
    plasmid,
    element,
    primer_tails=None,
    context_size=None,
    circular=True,
    allow_repeats=False,
    center_feature_name="tss",
    variant_feature_name="variant",
    name=None,
)
```

| Argument | Purpose |
| --- | --- |
| `plasmid` | Prefer a `PlasmidRecord`. A legacy callable `(fragment, element) -> sequence` is also accepted. |
| `element` | Element label used for primer-tail lookup/context routing. |
| `primer_tails` | Required when `plasmid` is a `PlasmidRecord`; ignored by the legacy callable branch. |
| `context_size`, `circular`, `allow_repeats` | Passed to `PlasmidRecord.plasmid_context()`. |
| `center_feature_name` | Added at the midpoint only when the legacy callable branch is used. `PlasmidRecord` results already receive `reporter_tss`. |
| `variant_feature_name` | Feature name/type used to detect or create the changed interval on REF/ALT fragments. |
| `name` | Fallback context/result metadata name. A `build(name=...)` argument takes precedence. |

#### `build(variant, *, genome=None, name=None) -> SequencePair`

The method recovers REF/ALT fragments from `Variant.from_sequences()` metadata
when present, including serialized input features. Otherwise it uses variant
alleles. It ensures each fragment has a variant feature, applies the
`PlasmidRecord` or legacy callable to both alleles, adds context-name metadata,
and returns a `SequencePair`.

`genome` is accepted for `Context` signature compatibility but is not read.

## Module-level utilities

### `infer_table18_primer_tails(...)`

Public subpackage/root export:

```python
infer_table18_primer_tails(
    fasta_reference_path,
    *,
    table18_primers,
    locations,
    flank=100,
    min_match=12,
    edge_slop=25,
    return_details=False,
)
```

| Argument | Purpose |
| --- | --- |
| `fasta_reference_path` | Indexed reference FASTA opened with `pysam`. |
| `table18_primers` | Mapping from element to `(forward_primer, reverse_primer)`. |
| `locations` | Matching mapping from element to `(chromosome, 1-based start, 1-based end)`. Every primer key must have a location. |
| `flank` | Extra bases fetched on both sides and expected primer-match position within the fetched sequence. |
| `min_match` | Minimum suffix length that must anneal to the target. |
| `edge_slop` | Distance within which a match is preferred as near the expected edge. |
| `return_details` | False returns tail tuples. True returns match dictionaries including orientation, annealing sequences, lengths, positions, and edge distance. |

### Advanced module-only helpers

These are not re-exported by `dna.__init__`, but are callable from
`gena_expression.dna.plasmid`:

| Function | Purpose |
| --- | --- |
| `reverse_complement(seq)` | Reverse-complement a DNA string. |
| `make_annotated_mcs_insert_fn(left_tail, reverse_primer_tail, primer_match=None, primer_element=None)` | Return an annotation-preserving circular-plasmid MCS replacement callable. |
| `centered_plasmid_window(sequence, *, center, size, circular, fill_sequence=None, fill_center=None)` | Extract an annotated linear/circular window, optionally filling bases outside one edited copy from another plasmid. |

Private names beginning with `_` and the internal `_GtfRecord`,
`_InMemoryAnnotation`, and `_TabixGtfAnnotation` support parsing, indexing, and
coordinate operations. They are implementation details rather than stable API
classes.

## Example

```python
from gena_expression import Genome, GenomeContext, Variant

genome = Genome(
    "reference.fa",
    annotation="annotation.gtf.gz",
    build="custom",
)
variant = Variant("chr1", 999, "A", "G", id="example")
pair = GenomeContext(length=2048).build(
    variant,
    genome=genome,
    name="example",
)

pair.ref.feature("variant")
pair.alt.feature("variant")
genome.close()
```

# SOX2 saturation-mutagenesis reproducibility

## Purpose

This document records the SOX2 validation experiments run while comparing the
genome-wide TSS saturation-mutagenesis runner with
[`in_silico_mutagenesis.ipynb`](../../api/notebooks/in_silico_mutagenesis.ipynb).
The notebook plot uses promoter-relative coordinates `0..599`; the equivalent
TSS-relative coordinates are `-600..-1`:

```text
notebook_position = tss_relative_position + 600
```

For example, notebook position 100 is approximately 500 bp upstream of the TSS.

## Revision caveat

All three experiments below were run from a working tree based on GitHub commit
[`d56f6e977c62bb96b066489aeb9ad615c9615775`](https://github.com/AIRI-Institute/GENA_LM/commit/d56f6e977c62bb96b066489aeb9ad615c9615775).
The saturation-mutagenesis implementation and plotting changes were uncommitted
when the experiments ran. Therefore `d56f6e9` is the committed baseline, **not
an exact source snapshot of the experiment code**. An exact GitHub commit link
does not exist for these completed runs. Their HDF5 provenance records input
file hashes, checkpoint hashes, model settings, description hashes, and scoring
definitions.

## Experiment 1 — `sox2_model_010726_liver_tss_window`

- Purpose: first end-to-end SOX2 smoke test.
- Code: [`run_saturation_mutagenesis.py`](run_saturation_mutagenesis.py) and
  [`saturation_mutagenesis.py`](saturation_mutagenesis.py).
- Git baseline: [`d56f6e9`](https://github.com/AIRI-Institute/GENA_LM/commit/d56f6e977c62bb96b066489aeb9ad615c9615775),
  with uncommitted experiment code.
- Checkpoint: `model_010726` / `expression_model_v1-1`, SHA-256
  `fffd4dabc782dbcee7c87a4b56e2c2f554b8bc1fdf2e0b7861cca4de7e752732`.
- Condition: adult liver ATAC-seq, `ENCFF556YQA`.
- Catalog TSS used: `chr3:181711924`, interpreted as 1-based.
- Mutation interval: TSS ±1000 bp.
- Scoring: fixed TSS window `[TSS-500, TSS+501)`, ATAC channel 0 weighted sum,
  alternative minus reference.
- DNA input: 1022 total tokens, comprising 510 upstream DNA tokens, 510
  downstream DNA tokens, CLS, and SEP.
- Orientations: forward and reverse complement.
- Runtime: 1843.20 seconds; 3.30 variants/second.
- HDF5: `sox2_saturation_mutagenesis.h5` (local, ignored by Git).
- Picture: [`sox2_forward_promoter_ref_alt.png`](sox2_forward_promoter_ref_alt.png).

This experiment was not expected to reproduce the notebook because both the
checkpoint and biological condition differed, and its scoring window was fixed
at the TSS.

## Experiment 2 — `sox2_model_270526_h1_tss_window`

- Purpose: switch to the requested `all_datasets` alias and notebook H1
  condition while retaining fixed TSS-centered scoring.
- Code: [`run_saturation_mutagenesis.py`](run_saturation_mutagenesis.py),
  [`saturation_mutagenesis.py`](saturation_mutagenesis.py), and
  [`sox2_inference_config.yaml`](sox2_inference_config.yaml).
- Git baseline: [`d56f6e9`](https://github.com/AIRI-Institute/GENA_LM/commit/d56f6e977c62bb96b066489aeb9ad615c9615775),
  with uncommitted experiment code.
- Configured model alias: `model_270526` / `all_datasets`.
- Actual checkpoint target during the run: `expression_model_v1-0`, SHA-256
  `8ba57538b916ecc9d4d16b6a591cd32bac4b0afc3bea60c27dc8c3be3d57aeb3`.
- Condition: H1 polyA-plus RNA-seq, `ENCFF081FQX`, matching the notebook.
- Catalog TSS used: `chr3:181711924`, interpreted as 1-based.
- Mutation interval: TSS ±600 bp.
- Scoring: fixed TSS window `[TSS-500, TSS+501)`.
- DNA input: 1022 total tokens; 510 upstream and 510 downstream DNA tokens.
- Orientations: forward and reverse complement.
- Runtime: 1083.97 seconds; 3.41 variants/second.
- HDF5: `sox2_model_270526_h1.h5` (local, ignored by Git).
- Picture: [`sox2_model_270526_h1_forward.png`](sox2_model_270526_h1_forward.png).

The biological condition now matched, but scoring was still different from the
notebook and the configured checkpoint was not the notebook's v1-2 checkpoint.

## Experiment 3 — `sox2_model_270526_h1_variant_window`

- Purpose: reproduce the notebook's variant-centered ATAC scoring window.
- Code: [`run_variant_centered_mutagenesis.py`](run_variant_centered_mutagenesis.py)
  plus [`saturation_mutagenesis.py`](saturation_mutagenesis.py).
- Git baseline: [`d56f6e9`](https://github.com/AIRI-Institute/GENA_LM/commit/d56f6e977c62bb96b066489aeb9ad615c9615775),
  with uncommitted experiment code.
- Actual checkpoint: `expression_model_v1-0`, SHA-256
  `8ba57538b916ecc9d4d16b6a591cd32bac4b0afc3bea60c27dc8c3be3d57aeb3`.
- Condition: notebook H1 description, `ENCFF081FQX`.
- Catalog TSS used: `chr3:181711924`, interpreted as 1-based.
- Mutation interval: TSS ±600 bp.
- Scoring: 501-bp ATAC window centered separately at every variant, alternative
  minus its position-specific reference.
- Model tokenization center: TSS, not variant.
- Pair execution: position-specific references and alternatives were inferred
  in separate model batches, not the notebook's joint `SequencePair` batches.
- DNA input: 1022 total tokens; 510 upstream and 510 downstream DNA tokens.
- Orientations: forward and reverse complement.
- Runtime: 1892.76 seconds; 1.93 variants/second.
- HDF5: `sox2_model_270526_h1_variant_centered.h5` (local, ignored by Git).
- Picture:
  [`sox2_model_270526_h1_variant_centered_forward.png`](sox2_model_270526_h1_variant_centered_forward.png).

The scale became substantially closer to the notebook: the forward range was
`-415.66..306.09`, and 95% of absolute effects were below 157.46. The strongest
effects remained near TSS-relative positions `-219..-212` (notebook positions
`381..388`), whereas the notebook's strongest region appears around notebook
position ~130 (approximately 470 bp upstream).

## Confirmed remaining differences

### 1. Completed Experiments 2–3 used v1-0, not v1-2

During Experiments 2–3, the local symlink resolved to:

```text
/workspace-SR003.nfs2/estsoi/CAGI5_benchmark/models/expression_model_v1-0/pytorch_model.bin
```

The notebook explicitly uses `expression_model_v1-2/pytorch_model.bin`. The two
files are different:

```text
expression_model_v1-0  8ba57538b916ecc9d4d16b6a591cd32bac4b0afc3bea60c27dc8c3be3d57aeb3
expression_model_v1-2  a089bb2e32d2e4f65200ebbbb4de51a46e7ec15c3ea43a275ec88f715cbfe2da
```

After documenting those experiments, the machine-local `model_270526` symlink
was corrected and now resolves to:

```text
/workspace-SR003.nfs2/estsoi/CAGI5_benchmark/models/expression_model_v1-2/pytorch_model.bin
```

### 2. DNA token counts are not equivalent

`CenteredTokenizer` reserves two positions for CLS and SEP internally.
Therefore:

```text
Notebook: dna_max_seq_len=1024 -> 1022 DNA tokens -> 511 upstream + 511 downstream
Runner:   dna_max_seq_len=1022 -> 1020 DNA tokens -> 510 upstream + 510 downstream
```

An exact notebook comparison must use `--dna-input-seq-len 1024
--num-before 511`.

### 3. Sequence bounds change the model input

Centered tokenization does not retrieve missing sequence beyond the supplied
`AnnotatedSequence`. It tokenizes only `sequence_text[center:]` downstream and
then keeps as many downstream tokens as are available.

The notebook sequence ends at `TES=181714436`, only 2512 bp downstream of its
TSS. Direct tokenization produced:

```text
                         Notebook       Experiment 3
Annotated sequence bp    12,732         20,400
DNA tokens used          937            1,020
Upstream DNA tokens      511            510
Downstream DNA tokens    426            510
Covered hg38 interval    [181708798,    [181708808,
                         181714436)      181714958)
```

Thus reproducing the notebook's `start=TSS-10220, end=TES` bounds changes the
actual model input: the notebook is downstream-context limited and supplies 83
fewer total DNA tokens than Experiment 3.

### 4. SOX2 coordinate convention was one base off

The notebook treats `181711924` as a 0-based center. The benchmark catalog is
1-based, so the equivalent catalog coordinate is `181711925`. The checked-in
SOX2 test row has now been corrected to 181711925. All completed experiments
above used the earlier value 181711924; their artifacts were not rewritten.

### 5. Joint pair execution remains to be reproduced

The notebook uses `pair_execution="joint"`, interleaving each reference and
alternative in the same forward batch and scoring the resulting
`PairPrediction`. Experiment 3 computes the same SNV window semantics using
separate absolute predictions and subtraction. Since the model is in evaluation
mode and SNV coordinate mapping is the identity, this difference should be
smaller than the checkpoint and sequence-context differences, but it remains
necessary for an exact workflow comparison.

## Verified matches

- H1 JSON: `ENCFF081FQX`.
- Rendered H1 description: API `metadata_to_description` and
  `ExpressionDataset.make_description_from_json` are byte-identical.
- DNA tokenizer: the notebook-local and Hugging Face `tokenizer.json` files
  have identical SHA-256
  `440cec76fcec5288d10f70b64f3f59289fe313dd1b6fc42bcf0fac319297dc3d`.
- Variant scorer in Experiment 3: ATAC channel 0, weighted sum, 501 bp centered
  at the variant, alternative minus reference.
- Displayed orientation: forward.

## Proposed exact-reproduction experiment

Before running another GPU profile:

1. Use the corrected `models/model_270526/pytorch_model.bin` symlink, which now
   points to the actual `expression_model_v1-2/pytorch_model.bin`.
2. Use the corrected 1-based SOX2 coordinate `181711925`.
3. Use `dna_input_seq_len=1024` and `num_before=511`.
4. Construct the same bounded sequence as the notebook:
   `[TSS-10220, TES=181714436)` in 0-based hg38 coordinates.
5. Generate only the 600-bp upstream promoter substitutions needed for the
   comparison.
6. Infer and score through joint `SequencePair` execution.
7. Save outputs under a new experiment name; do not overwrite Experiments 1–3.

## Experiment 4 — `sox2_dev_loss_h1_notebook_exact`

- Purpose: execute the notebook workflow with all previously identified
  differences removed.
- Code:
  [`run_sox2_notebook_reproduction.py`](run_sox2_notebook_reproduction.py).
- Git baseline: [`d56f6e9`](https://github.com/AIRI-Institute/GENA_LM/commit/d56f6e977c62bb96b066489aeb9ad615c9615775),
  with uncommitted experiment code; no exact GitHub commit exists yet.
- Semantic model: `dev_loss`.
- Checkpoint: `expression_model_v1-2`, MD5
  `02668c3e5572c58f1322b0faba22d7d7`, SHA-256
  `a089bb2e32d2e4f65200ebbbb4de51a46e7ec15c3ea43a275ec88f715cbfe2da`.
- Condition: notebook H1 description, `ENCFF081FQX`.
- TSS: `chr3:181711924` 0-based / `chr3:181711925` 1-based.
- Sequence: exact notebook interval `[181701704, 181714436)`, ending at the
  SOX2 TES.
- DNA input: 1024 total positions, `num_before=511`.
- Mutation region: the exact 600-bp upstream promoter `[9620,10220)` in the
  local sequence.
- Variants: 1800 substitutions and 20 reproducibly sampled deletions.
- Scoring: ATAC channel 0 weighted sum over 501 bp centered at each variant,
  alternative minus reference.
- Execution: joint `SequencePair` batches, at most 200 pairs per forward.
- Orientation: notebook forward orientation.
- Runtime: 126.29 seconds; 14.41 variants/second.
- Substitution range: `-202.13..138.70`; 95% of absolute substitution effects
  are below 76.74.
- Strongest substitution: promoter position 133, `A>C`, score `-202.13`.
- Results:
  [`sox2_notebook_reproduction.parquet`](sox2_dev_loss_h1_notebook_exact_output/sox2_notebook_reproduction.parquet).
- Provenance:
  [`sox2_notebook_reproduction_provenance.json`](sox2_dev_loss_h1_notebook_exact_output/sox2_notebook_reproduction_provenance.json).
- Picture:
  [`sox2_notebook_reproduction_track_window.png`](sox2_dev_loss_h1_notebook_exact_output/sox2_notebook_reproduction_track_window.png).

This experiment reproduces the notebook's characteristic pattern: the dominant
negative-effect cluster occurs at promoter positions approximately 128–144,
including a minimum near -200, with secondary clusters near positions 330,
380, and 490. The earlier shift toward positions 381–388 was therefore driven
by the combined checkpoint, sequence-bound, token-budget, coordinate, and
execution-path differences rather than by the plotting helper.

## Local semantic model aliases

The aliases relevant to these experiments are now:

| Local name | External checkpoint | MD5 | Run name |
|---|---|---|---|
| `models/model_270526` (`all_datasets`) | `expression_model_v1-0` | `8dc611f37c6cecd34a1ff341323c409f` | `s3://genalm/expr/runs/model_270526/` |
| `models/model_010726` (`all_datasets_2`) | `expression_model_v1-1` | `aea878eefd1338d1cf5f472243e84881` | `s3://genalm/expr/runs/model_010726/` |
| `models/dev_loss` | `expression_model_v1-2` | `02668c3e5572c58f1322b0faba22d7d7` | `s3://genalm/expr/runs/devloss_110726/` |

The `devloss_110726` S3 prefix currently contains the checkpoint and its MD5
file but no YAML. `models/dev_loss/inference_example_config.yaml` was therefore
copied from the exact external configuration referenced by the notebook, with
only the two backbone paths changed to the repository-local persistent model
aliases.

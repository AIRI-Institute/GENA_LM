# E2-selected candidate TSSs

This directory contains the TSS-selection pass from experiment E2 for all
1,581 benchmark samples. No variant filtering, distance scoring, or
gene prediction is performed here.

## Selection rule

GENCODE release 50 transcript TSSs are restricted to the gene universe in
`human_all.tsv` and filtered independently for each gene:

1. If one or more transcript TSSs overlap an ATAC peak within ±100 bp,
   keep only those transcripts; otherwise retain all transcript TSSs.
2. Prefer MANE Select transcripts when present. Otherwise prefer the best
   lowest-numbered transcript support level, following the E2 tie behavior.
3. If multiple genomic TSS coordinates remain, retain all of them.
4. If a gene is present in `human_all.tsv` but has no GENCODE 50 transcript,
   use its single original `human_all.tsv` TSS as a fallback.

Coordinates are hg38 and 1-based. `annotation_source` identifies whether the
coordinate and strand come from the GENCODE GTF or the `human_all.tsv`
fallback.

## Files

- `selected_tss_catalog.tsv`: one row per unique selected gene/TSS coordinate.
  This is the recommended table to annotate externally. `tss_id` is the join
  key. Transcript fields describe transcripts remaining at that coordinate.
- `sample_candidate_tss.tsv`: one row per sample/candidate/selected-TSS
  relationship. Join external annotations back through `tss_id`.
- `unmapped_candidate_genes.tsv`: candidate genes with no exportable E2 TSS,
  including their occurrence counts and the reason.

`is_ground_truth` is included only as mapping metadata and is not used during
TSS selection.

## Coverage

- Samples: 1,581
- Candidate-gene occurrences: 54,372
- Candidate occurrences with at least one selected TSS:
  39,528
- Unique candidate genes: 15,054
- Unique candidate genes with selected TSSs: 11,080
- Unique selected gene/TSS coordinates: 12,165
- Sample/candidate/TSS mapping rows: 43,009
- Samples whose ground-truth gene has at least one selected TSS:
  1,128/1,581
- Ground-truth samples without an exportable TSS: 453
  absent from `human_all.tsv`; 0 present in
  `human_all.tsv` but still unavailable after applying the fallback
- Unique candidate genes without an exportable TSS: 3,974

## Reproduction

From the repository root:

```bash
python scripts/export_e2_selected_tss.py
```

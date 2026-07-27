# Benchmark Comparison Notebooks

These notebooks calculate correlations and cell-type specificity scores for the
GENA_LM and AlphaGenome expression benchmark.

## Notebooks

- `notebooks/gena_14_celllines_compare.ipynb` compares one GENA_LM prediction
  file against the 14-cell GT CSV (`test_true_human.csv` or `valid_true_human.csv`).
- `notebooks/gena_all_celllines_compare.ipynb` compares GENA_LM predictions
  against the full qnorm GT matrix. It transforms GT from cell x gene to
  gene x cell before scoring.
- `notebooks/alphagenome_14_celllines_compare.ipynb` compares one AlphaGenome
  14-cell prediction file with GT.
- `notebooks/gena_alphagenome_ontology_compare.ipynb` averages all cell IDs
  belonging to the same ontology, then compares GENA_LM and AlphaGenome with
  log2(TPM + 1) GT.

## Main Metrics

- `corr_genes`: for each cell type, correlation across genes; then mean over
  cell types.
- `corr_cells`: for each gene, correlation across cell types; then mean over
  genes.
- `deviation_r`: cell-type specificity score from `score_ct_specificity.py`.

## Expected Data Paths On Anogena

```text
/home/jovyan/dpanc/benchmarking/GENA_LM/predictions_results
/home/jovyan/dpanc/benchmarking/AlphaGenome/predictions
/home/jovyan/dpanc/benchmarking/data
```

Edit the first parameter cell in each notebook to change model, split, cell set,
or prediction paths.

## Git Notes

These notebooks were copied from the working analysis notebooks and cleaned for
git: outputs and execution counts were removed, paths were grouped at the top,
and fragile imports were replaced with repo-relative imports.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from score_ct_specificity import score_predictions


TRUE_PATH = Path("/home/biophysinf/DashaP/benchmarking/AlphaGenome/data/borzoi_all_ids_qnorm_matrix.csv")
PRED_PATH = Path("/scratch/biophysinf-alphagenome/alphagenome_benchmark_results_02062026/alphagenome_predictions_test_intervals.tsv")
SELECTED_TARGETS = Path("/home/biophysinf/DashaP/benchmarking/data/selected_targets.csv")
OUT_PATH = Path("/home/biophysinf/DashaP/benchmarking/AlphaGenome/benchmark_outputs/simple_alphagenome_benchmark_summary.csv")

split_name = "test"
prediction_functions_name = "intervals"
cfg = {"path_to_selected": str(SELECTED_TARGETS)}


def align_dataframes(true, pred):
    true_subset = true[true["gene_id"].isin(pred["gene_id"])].reset_index(drop=True)
    pred_subset = pred[pred["gene_id"].isin(true_subset["gene_id"])].reset_index(drop=True)
    true_sorted = true_subset.sort_values("gene_id")
    pred_sorted = pred_subset.sort_values("gene_id")
    common_cols = sorted(set(true.columns[1:]).intersection(pred.columns[1:]))
    true_aligned = true_sorted[["gene_id"] + common_cols]
    pred_aligned = pred_sorted[["gene_id"] + common_cols]
    return true_aligned, pred_aligned


def load_true_df(path):
    true_df = pd.read_csv(path)
    if "gene_id" in true_df.columns:
        return true_df

    gene_cols = [c for c in true_df.columns if str(c).startswith(("ENSG", "ENSMUSG"))]
    true_df = (
        true_df.set_index("id")[gene_cols]
        .T
        .reset_index()
        .rename(columns={"index": "gene_id"})
    )
    return true_df


true_df = load_true_df(TRUE_PATH)
pred_df = pd.read_csv(PRED_PATH, sep=None, engine="python")

summary_rows = []

true_df, pred_df = align_dataframes(true_df, pred_df)
score_dict = score_predictions(true_df, pred_df, cfg.get("path_to_selected", "selected_targets.csv"), need_log=False)
deviation_r = score_dict.get("deviation_r", float("nan"))

true_aligned, pred_aligned = true_df, pred_df

cell_corrs = []
for i in range(true_aligned.shape[0]):
    true_vec = np.array(true_aligned.iloc[i, 1:].values, dtype=np.float64)
    pred_vec = np.array(pred_aligned.iloc[i, 1:].values, dtype=np.float64)
    if np.std(true_vec) == 0 or np.std(pred_vec) == 0:
        continue
    cell_corrs.append(np.corrcoef(true_vec, pred_vec)[0, 1])
avg_cell_corr = float(np.nan) if len(cell_corrs) == 0 else float(np.mean(cell_corrs))

gene_corrs = []
for j in range(1, true_aligned.shape[1]):
    true_vec = np.array(true_aligned.iloc[:, j].values, dtype=np.float64)
    pred_vec = np.array(pred_aligned.iloc[:, j].values, dtype=np.float64)
    if np.std(true_vec) == 0 or np.std(pred_vec) == 0:
        continue
    gene_corrs.append(np.corrcoef(true_vec, pred_vec)[0, 1])
avg_gene_corr = float(np.nan) if len(gene_corrs) == 0 else float(np.mean(gene_corrs))

summary_rows.append({
    "split": split_name,
    "prediction_function": prediction_functions_name,
    "avg_gene_corr": avg_gene_corr,
    "avg_cell_type_corr": avg_cell_corr,
    "deviation_r": deviation_r,
    "genes_evaluated": len(cell_corrs),
    "cell_types_evaluated": len(gene_corrs),
})

summary_df = pd.DataFrame(summary_rows)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(OUT_PATH, index=False)
print(summary_df.to_string(index=False))
print(f"Saved: {OUT_PATH}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script that (optionally) runs AlphaGenome predictions and compares them to truth.
Saves per-split summary (avg cell-type corr, avg gene corr, deviation_r) and prediction files.

Usage:
    python benchmark_with_prediction.py --config config.yaml
"""

import argparse
import os
import re
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO

import alphagenome
from alphagenome.data import genome
from alphagenome.models import dna_client, interval_scorers
from alphagenome.models import variant_scorers

from score_ct_specificity import score_predictions

# We manually compared ontology ids with Alphagenome ontology names using ENCODE
EXPERIMENT_ONTOLOGY_DICT = {
 'UBERON:0001159 total RNA-seq': 'ENCSR094GVZ',
 'UBERON:0001150 total RNA-seq': 'ENCSR586SYA',
 'UBERON:0002113 total RNA-seq': 'ENCSR892LBU',
 'UBERON:0002369 total RNA-seq': 'ENCSR763OMY',
 'UBERON:0002168 total RNA-seq': 'ENCSR045GTF',
 'UBERON:0001157 total RNA-seq': 'ENCFF672VYQ',
 'UBERON:0000945 polyA plus RNA-seq': 'ENCSR721HDG',
 'UBERON:0001264 polyA plus RNA-seq': 'ENCSR571BML',
 'EFO:0002713 total RNA-seq': 'ENCSR128CYL',
 'EFO:0001187 total RNA-seq': 'ENCSR245ATJ',
 'UBERON:0001264 total RNA-seq': 'ENCSR432EBE',
 'EFO:0001187 polyA plus RNA-seq': 'ENCSR561FEE',
 'UBERON:0001115 total RNA-seq': 'ENCSR357BYU',
 'UBERON:0000945 total RNA-seq': 'ENCSR471RUK'
}
ONTOLOGY_LIST = list(EXPERIMENT_ONTOLOGY_DICT.keys())

# -------------------------
# Helper utilities
# -------------------------
def build_split_path(base_save, split, prediction_functions_name):
    """Construct split-specific path from template or base filename."""
    if base_save is None:
        return None
 # if template contains placeholders, fill only those present
    if ("{split}" in base_save) and ("{prediction_functions_name}" in base_save):
        fmt_kwargs = {}
        fmt_kwargs["split"] = split
        fmt_kwargs["prediction_functions_name"] = prediction_functions_name
        return base_save.format(**fmt_kwargs)

    else:
        root, ext = os.path.splitext(base_save)
        return f"{root}_{split}{ext}"

# -------------------------
# Notebook-derived code
# -------------------------
def make_interval(row):
    gene_start = int(row["TSS"])
    gene_end = int(row["TSS"])
    chrom = row["chrom"]
    strand = row["gene_strand"]
    id_name = row["gene_id_unversioned"]

    interval = genome.Interval(
        chromosome=chrom,
        start=gene_start,
        end=gene_end,
        strand=strand,
        name=id_name,
    )

    interval = interval.resize(dna_client.SEQUENCE_LENGTH_16KB)
    return interval

def build_gene_intervals_dict(forward_df, reverse_df):
    gene_intervals = {}

    assert forward_df['gene_id_unversioned'].is_unique
    for _, row in forward_df.iterrows():
        gid = row['gene_id_unversioned']
        gene_intervals[gid] = make_interval(row)

    assert reverse_df['gene_id_unversioned'].is_unique
    for _, row in reverse_df.iterrows():
        gid = row['gene_id_unversioned']
        gene_intervals[gid] = make_interval(row)
    return gene_intervals

def get_gene_scores_intervals(gene_id, interval, model, ontology_list):
    """
    Use model.score_interval + GeneMaskScorer to get gene-level aggregated values.
    Returns pivot DataFrame (track_name x gene_id) or None on failure.
    """
    try:
        strand = interval.strand
        res = model.score_interval(
            interval,
            interval_scorers=[
                interval_scorers.GeneMaskScorer(
                    requested_output=dna_client.OutputType.RNA_SEQ,
                    width=10001, # if we use 1Mb set it as 200_001, if 16Kb use 10_001
                    aggregation_type=interval_scorers.IntervalAggregationType.MEAN,
                )
            ],
        )[0]

        mask = res.var["strand"] == strand
        filtered_names = res.var.loc[mask, "name"]
        df = pd.DataFrame(res.X[:, mask], index=res.obs["gene_id"], columns=filtered_names)

        cols = [c for c in df.columns if c in ontology_list]
        df = df[cols]

        regex = re.compile(f"^{gene_id}\\b")
        matched_genes = [g for g in df.index if regex.match(g)]
        if not matched_genes:
            # no match — return None (caller may skip)
            print(f"[WARN] gene {gene_id} not found in annData index")
            return None

        if len(matched_genes) != 1:
            raise ValueError(f"Ambiguous gene match for {gene_id}: found {len(matched_genes)} entries {matched_genes}")

        matched_gene = matched_genes[0]
        filtered_df = df.loc[[matched_gene]]
        pivot_df = filtered_df.T
        pivot_df.index.name = "track_name"
        pivot_df.columns = [gene_id]
        return pivot_df

    except Exception as e:
        print(f"[ERROR] Failed to process {gene_id}: {e}")
        return None


def get_gene_scores_variants(gene_id, interval, model, ref_genome, track_names):
    """
    Parameters
    ----------
    interval: genome.Interval
        Interval object
    model: dna_client.ModelVersion
        AlphaGenome model
    ref_genome: dict
        Biopython dictionary with chromosome sequences
    track_names: list
        List of tracks for filtering

    Returns
    -------
    pd.DataFrame
        DataFrame with gene scores
    """
    try:
        variant_chromosome = interval.chromosome
        variant_position = (interval.start + interval.end) // 2  # <- middle
        strand = interval.strand
        name = interval.name

        # Using the reference base as both reference and alternate for a neutral variant
        variant_reference_bases = ref_genome[variant_chromosome].seq[variant_position].upper()
        variant_alternate_bases = ref_genome[variant_chromosome].seq[variant_position].upper()

        variant = genome.Variant(
            chromosome=variant_chromosome,
            position=variant_position,
            reference_bases=variant_reference_bases,
            alternate_bases=variant_alternate_bases,
        )

        resized_interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_16KB)
        resized_interval.strand = strand
        resized_interval.name = name

        # Using score_variant to get variant scores
        variant_scores = model.score_variant(
            interval=resized_interval,
            variant=variant,
            variant_scorers=[variant_scorers.GeneMaskActiveScorer(
                requested_output=alphagenome.models.dna_output.OutputType(4)
            )],
            organism=dna_client.Organism.HOMO_SAPIENS
        )

        # tidy dataframe
        tidy_scores = variant_scorers.tidy_scores([variant_scores], match_gene_strand=True)
        filtered_scores = tidy_scores[tidy_scores["track_name"].isin(track_names)]
        filtered_scores = filtered_scores[filtered_scores["gene_id"] == name]
        filtered_scores = filtered_scores[filtered_scores["track_strand"] == strand]

        pivot_df = filtered_scores.pivot_table(
            index="track_name",
            columns="gene_id",
            values="raw_score"
        )

        return pivot_df

    except Exception as e:

        print(f"[ERROR] Failed to process {gene_id}: {e}")

        return None

def get_fold_intervals_for_intervals(model_version, api_key, gene_intervals, ref_genome, prediction_function="intervals"):
    """
    For given gene_intervals (dict gene_id->Interval) run predictions for each gene.
    Returns DataFrame with index = track_name (ENCSR...), columns = gene_ids (no versions).
    """
    if isinstance(model_version, str):
        try:
            model_enum = getattr(dna_client.ModelVersion, model_version)
        except Exception:
            raise ValueError("model_version string not found in dna_client.ModelVersion")
    else:
        model_enum = model_version

    ref_genome = SeqIO.to_dict(SeqIO.parse(ref_genome, "fasta"))

    model = dna_client.create(api_key=api_key, model_version=model_enum)

    all_results = []
    for name, interval in tqdm(gene_intervals.items(), desc="predict genes"):
        if prediction_function == "intervals":
            gene_scores = get_gene_scores_intervals(name, interval, model, ONTOLOGY_LIST)
            if gene_scores is not None:
                all_results.append(gene_scores)
        elif prediction_function == "variants":
            gene_scores = get_gene_scores_variants(name, interval, model, ref_genome, ONTOLOGY_LIST)
            if gene_scores is not None:
                all_results.append(gene_scores)

    if len(all_results) == 0:
        raise RuntimeError("No results produced by get_gene_scores_intervals")

    results_df = pd.concat(all_results, axis=1)
    results_df_mapped = results_df.rename(index=EXPERIMENT_ONTOLOGY_DICT)

    return results_df_mapped

# -------------------------
# Predictions wrapper
# -------------------------
def generate_predictions(cfg):
    api_key = os.environ.get("ALPHAGENOME_API_KEY") or cfg.get("api_key")
    if not api_key:
        raise ValueError("Set ALPHAGENOME_API_KEY in the environment before running predictions")
    model_version = cfg.get("model_version", "FOLD_1")
    which_split = cfg.get("which_split", "test")  # "test" or "valid"

    g_fwd_val = cfg.get("genes_forward_val", "human.valid.forward.csv")
    g_rev_val = cfg.get("genes_reverse_val", "human.valid.reverse.csv")
    g_fwd_test = cfg.get("genes_forward_test", "human.test.forward.csv")
    g_rev_test = cfg.get("genes_reverse_test", "human.test.reverse.csv")

    genes_forward_val = pd.read_csv(g_fwd_val, sep='\t')
    genes_reverse_val = pd.read_csv(g_rev_val, sep='\t')
    genes_forward_test = pd.read_csv(g_fwd_test, sep='\t')
    genes_reverse_test = pd.read_csv(g_rev_test, sep='\t')

    ref_genome = cfg.get("ref_genome", "hg38.fna")

    val_gene_intervals = build_gene_intervals_dict(genes_forward_val, genes_reverse_val)
    test_gene_intervals = build_gene_intervals_dict(genes_forward_test, genes_reverse_test)
    prediction_function = cfg.get("which_prediction_functions", "intervals") # "variants"

    gene_intervals = test_gene_intervals if which_split == "test" else val_gene_intervals

    results_df_mapped = get_fold_intervals_for_intervals(model_version, api_key, gene_intervals, ref_genome, prediction_function=prediction_function)
    return results_df_mapped

def load_and_normalize_predictions(pred_df,
                                   true_df,
                                   mapping_genes=None,
                                   mapping_tissues=None,
                                   save_path=None):

    # first_col = pred_df.columns[0]
    # if str(first_col).lower() in ("track_name", "track", "index"):
    #     pred_df = pred_df.set_index(first_col)
    pred_out = pred_df.T.reset_index().rename(columns={pred_df.T.reset_index().columns[0]: "gene_id"})
    pred_out.columns.name = None

    pred_out["gene_id"] = pred_out["gene_id"].astype(str)

    if mapping_genes:
        try:
            pred_out["gene_id"] = pred_out["gene_id"].replace(mapping_genes)
        except Exception as e:
            print("[WARN] failed to apply mapping_genes:", e)

    if mapping_tissues:
        try:
            pred_out = pred_out.rename(columns=mapping_tissues)
        except Exception as e:
            print("[WARN] failed to apply mapping_tissues:", e)

    if "gene_id" not in list(true_df.columns):
        raise ValueError("true_df must contain 'gene_id' column")
    target_sample_cols = [c for c in true_df.columns if c != "gene_id"]

    missing = [c for c in target_sample_cols if c not in pred_out.columns]
    if missing:
        print(f"[WARN] {len(missing)} target columns missing in predictions; filling with NaN. Example missing: {missing[:10]}")
        for c in missing:
            pred_out[c] = pd.NA

    cols_to_keep = ["gene_id"] + target_sample_cols
    pred_df_norm = pred_out[cols_to_keep].copy()

    if pred_df_norm["gene_id"].duplicated().any():
        print("[WARN] duplicated gene_id in predictions (after normalization). Duplicates will remain in output.")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pred_df_norm.to_csv(save_path, index=False)

    return pred_df_norm

### Aligning 2 df for correlations
def align_dataframes(true, pred):
    true_subset = true[true['gene_id'].isin(pred['gene_id'])].reset_index(drop=True)
    pred_subset = pred[pred['gene_id'].isin(true_subset['gene_id'])].reset_index(drop=True)
    true_sorted = true_subset.sort_values('gene_id')
    pred_sorted = pred_subset.sort_values('gene_id')
    common_cols = sorted(set(true.columns[1:]).intersection(pred.columns[1:]))
    true_aligned = true_sorted[['gene_id'] + common_cols]
    pred_aligned = pred_sorted[['gene_id'] + common_cols]
    return true_aligned, pred_aligned

# -------------------------
# Main orchestration
# -------------------------
def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    outdir = cfg.get("output_dir", "./out")
    os.makedirs(outdir, exist_ok=True)

    true_test_file = cfg.get("true_test_file", "test_true_human.csv")
    true_valid_file = cfg.get("true_valid_file", "valid_true_human.csv")
    splits = cfg.get("splits", ["test", "valid"])
    summary_rows = []
    prediction_function = cfg.get("prediction_functions", ["intervals", "variants"])

    for prediction_functions_name in prediction_function:
        for split_name in splits:
            print(f"\n=== Processing split: {split_name} with {prediction_functions_name} function ===")

            true_path = true_test_file if split_name == "test" else true_valid_file
            if not os.path.exists(true_path):
                print(f"[ERROR] true file not found: {true_path} — skipping split {split_name}")
                continue
            true_df = pd.read_csv(true_path)

            # build split-specific cfg for prediction
            cfg_gen = dict(cfg)
            cfg_gen["which_split"] = split_name
            cfg_gen["which_prediction_functions"] = prediction_functions_name

            base_save = cfg.get("save_pred_path")
            if base_save:
                cfg_gen["save_pred_path"] = build_split_path(base_save, split_name, prediction_functions_name)

            # produce or load predictions
            if cfg.get("run_prediction", False):
                print("Running AlphaGenome predictions for split:", split_name)
                pred_matrix = generate_predictions(cfg_gen)  # index=track_name, columns=gene_ids

                if os.path.exists(cfg.get("genes_forward_val", "human.valid.forward.csv")) and os.path.exists(cfg.get("genes_forward_test", "human.test.forward.csv")):
                    genes_forward_val = pd.read_csv(cfg.get("genes_forward_val", "human.valid.forward.csv"), sep='\t')
                    genes_reverse_val = pd.read_csv(cfg.get("genes_reverse_val", "human.valid.reverse.csv"), sep='\t')
                    genes_forward_test = pd.read_csv(cfg.get("genes_forward_test", "human.test.forward.csv"), sep='\t')
                    genes_reverse_test = pd.read_csv(cfg.get("genes_reverse_test", "human.test.reverse.csv"), sep='\t')

                    mapping1 = dict(zip(genes_forward_val['gene_id_unversioned'], genes_forward_val['gene_id']))
                    mapping2 = dict(zip(genes_reverse_val['gene_id_unversioned'], genes_reverse_val['gene_id']))
                    mapping3 = dict(zip(genes_forward_test['gene_id_unversioned'], genes_forward_test['gene_id']))
                    mapping4 = dict(zip(genes_reverse_test['gene_id_unversioned'], genes_reverse_test['gene_id']))
                    mapping_genes = {**mapping1, **mapping2, **mapping3, **mapping4}

                    tissues_file = cfg.get("tissues", "Expression_dataset_v1_csv_file_mappings_qnorm.csv")
                    tissues = pd.read_csv(tissues_file)
                    mapping_tissues = dict(zip(tissues['original_id'], tissues['id']))

                if cfg_gen.get("save_pred_path"):
                    save_path = os.path.join(outdir, cfg_gen.get("save_pred_path"))

                pred_df = load_and_normalize_predictions(pred_matrix,
                                    true_df,
                                    mapping_genes=mapping_genes,
                                    mapping_tissues=mapping_tissues, save_path=save_path)

            else:
                # load split-specific file derived from save_pred_path
                base_save = cfg_gen.get("save_pred_path")
                if base_save is None:
                    raise ValueError("No 'save_pred_path' in config and run_prediction is False. Provide save_pred_path template or enable run_prediction.")

                if not os.path.exists(base_save):
                    raise FileNotFoundError(f"Expected prediction file for split '{split_name}' not found.")
                pred_df = pd.read_csv(base_save, index_col=None)

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

            print(f"{split_name}: avg_cell_corr={avg_cell_corr:.4f}, avg_gene_corr={avg_gene_corr:.4f}, deviation_r={deviation_r:.4f}")


    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(outdir, "Alphagenome_benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Saved benchmark summary to", summary_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)

import pandas as pd
import numpy as np
import argparse
import sys

from functools import lru_cache

def get_genes_df(selected_targets):
    # Split comma-separated gene IDs into individual rows
    gene_ids = selected_targets["gene_id"].str.split(",")
    flattened_gene_ids = sum(gene_ids, [])

    # Create matching assembly values by repeating each assembly 
    # based on number of genes for that row
    assembly_values = []
    for _, row in selected_targets.iterrows():
        num_genes = row["gene_id"].count(",") + 1
        assembly_values.extend([row["assembly"]] * num_genes)

    # Create DataFrame with gene IDs and corresponding assemblies
    selected_genes = pd.DataFrame({
        "gene_id": flattened_gene_ids,
        "assembly": assembly_values
    })

    return selected_genes

def get_targets_df(selected_targets):
    # Split comma-separated tragets into individual rows
    target_ids = selected_targets["on_targets"].str.split(",") + selected_targets["off_targets"].str.split(",")
    flattened_target_ids = sum(target_ids, [])

    # Create matching assembly values by repeating each assembly 
    # based on number of cell types for that row
    assembly_values = []
    for _, row in selected_targets.iterrows():
        num_cell_types = row["on_targets"].count(",") + row["off_targets"].count(",") + 2
        assembly_values.extend([row["assembly"]] * num_cell_types)

    # Create DataFrame with gene IDs and corresponding assemblies
    selected_targets = pd.DataFrame({
        "target_id": flattened_target_ids,
        "assembly": assembly_values
    })
    return selected_targets

def compute_deviation(X, Y, pseudo_count=0.01, need_log=True):
    if need_log:
        return np.log2((X + pseudo_count) / np.multiply(X + pseudo_count, Y + pseudo_count) ** 0.5)
    else:
        return X - 0.5 * (X + Y)    

def fake_predictions(ground_truth, std=0.1):
    gene_mean = ground_truth.apply(
        lambda row: np.mean([v for k, v in row.items() if (k.startswith("ENCFF") or k.startswith("McKellar")) and pd.notna(v)]),
        axis=1
    )
    predictions = ground_truth.copy()
    for column in predictions.columns:
        if column != "gene_id":
            predictions.loc[:, column] = np.abs(gene_mean + np.random.normal(0, std, size=len(predictions)))
    return predictions


def cached_read(path: str):
    @lru_cache
    def cached_read_impl(df_path: str):
        return pd.read_csv(df_path)
    return cached_read_impl(path).copy()

def score_predictions(ground_truth, predictions, 
                    path_to_selected, 
                    unversion_gene_ids=False, 
                    need_log=True, 
                    pseudo_count=0.01, 
                    logger=None,
                    return_big_data=False,
                    ):
    """Score predictions against ground truth data."""

    # Load data if paths provided
    if isinstance(ground_truth, str):
        ground_truth = cached_read(ground_truth)
    if isinstance(predictions, str):
        predictions = cached_read(predictions).copy()

    # Load and process selected genes and targets
    selected = cached_read(path_to_selected).copy()
    selected_genes = get_genes_df(selected).drop_duplicates(subset=["gene_id"])
    selected_targets = get_targets_df(selected).drop_duplicates(subset=["target_id"])

    if unversion_gene_ids:
        ground_truth["gene_id"] = ground_truth["gene_id"].str.split(".").str[0]
        predictions["gene_id"] = predictions["gene_id"].str.split(".").str[0]
        selected_genes["gene_id"] = selected_genes["gene_id"].str.split(".").str[0]

    # Select genes and targets
    assert ground_truth["gene_id"].duplicated().sum() == 0, "Ground truth contains duplicate gene IDs"
    assert predictions["gene_id"].duplicated().sum() == 0, "Predictions contain duplicate gene IDs"

    genes_set = set(ground_truth["gene_id"].values) & set(predictions["gene_id"].values) & set(selected_genes["gene_id"].values)
    ground_truth = ground_truth[ground_truth["gene_id"].isin(genes_set)]
    predictions = predictions[predictions["gene_id"].isin(genes_set)]

    # validate data
    if len(ground_truth) == 0 or len(predictions) == 0:
        results = {"genes_number": np.nan, "ct_number": np.nan, "expression_pairs": np.nan, "deviation_r": np.nan}
        return results

    assert len(ground_truth) == len(predictions), "Ground truth and predictions have different number of genes"

    # define genome: if gene_id starts with ENSG, it's human, if with ENSMUSG, it's mouse
    def get_genome(gene_id):
        if gene_id.startswith("ENSG"):
            return "human"
        elif gene_id.startswith("ENSMUSG"):
            return "mouse"
        else:
            raise ValueError(f"Unknown gene ID: {gene_id}")

    ground_truth_genome = ground_truth["gene_id"].apply(get_genome)
    predictions_genome = predictions["gene_id"].apply(get_genome)

    # check that genome is unique
    assert ground_truth_genome.nunique() == 1, "Ground truth contains both human and mouse genes"
    assert predictions_genome.nunique() == 1, "Predictions contain both human and mouse genes"
    assert ground_truth_genome.iloc[0] == predictions_genome.iloc[0], "Ground truth and predictions contain different genomes"
    genome = ground_truth_genome.iloc[0]

    # select columns from selected_targets that match genome
    if genome == "human":
        prefix = "ENCFF"
    elif genome == "mouse":
        prefix = "McKellar"
    else:
        raise ValueError(f"Unknown genome: {genome}")

    columns = [c for c in selected_targets["target_id"].values if c.startswith(prefix)]
    # Check that all selected columns exist in ground truth and predictions
    missing_in_ground_truth = set(columns) - set(ground_truth.columns.values)
    missing_in_predictions = set(columns) - set(predictions.columns.values)
    
    if len(missing_in_ground_truth) > 0 or len(missing_in_predictions) > 0:
        message = f"Ground truth or predictions missing columns: {missing_in_ground_truth} {missing_in_predictions}, will skip cell type specificity scoring"
        if logger:
            logger.debug(message)
        else:
            print(message)
        results = {"genes_number": np.nan, "ct_number": np.nan, "expression_pairs": np.nan, "deviation_r": np.nan}
        return results

    ground_truth = ground_truth.set_index("gene_id")[columns]
    predictions = predictions.set_index("gene_id")[columns]

    # Validate data
    # on train, we might have NaN values in ground truth and predictions
    # this happens because with n_keys option, we take only subset of cell types for each gene
    # thus we might not have all cell_type*gene combinations in ground truth and predictions
    # we will drop these genes from scoring
    if pd.isna(ground_truth).sum().sum() > 0 or pd.isna(predictions).sum().sum() > 0:
        ground_truth = ground_truth.dropna()
        predictions = predictions.dropna()
        if len(ground_truth) == 0 or len(predictions) == 0:
            results = {"genes_number": np.nan, "ct_number": np.nan, "expression_pairs": np.nan, "deviation_r": np.nan}
            return results
        # assert pd.isna(ground_truth).sum().sum() == 0, "Ground truth contains NaN values: \n" + ground_truth[pd.isna(ground_truth)].head().to_string()
        # assert pd.isna(predictions).sum().sum() == 0, "Predictions contain NaN values: \n" + predictions[pd.isna(predictions)].head().to_string()

    # Calculate metrics
    results = {}
    results["genes_number"] = len(predictions)
    results["ct_number"] = len(predictions.columns)
    results["expression_pairs"] = len(columns) * (len(columns) - 1) / 2

    deviation_pred = []
    deviation_true = []
    targets_pairs = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            deviation_pred.extend(compute_deviation(
                predictions[columns[i]].values,
                predictions[columns[j]].values,
                pseudo_count=pseudo_count,
                need_log=need_log
            ))
            deviation_true.extend(compute_deviation(
                ground_truth[columns[i]].values,
                ground_truth[columns[j]].values,
                pseudo_count=pseudo_count,
                need_log=need_log
            ))
            targets_pairs.extend([(columns[i], columns[j])] * len(predictions))

    assert np.std(deviation_pred) != 0, "std(deviation_pred) is 0"
    results["deviation_r"] = np.corrcoef(deviation_pred, deviation_true)[0, 1]

    if return_big_data:
        results["big_data"] = {}
        results["big_data"]["deviation_pred"] = deviation_pred
        results["big_data"]["deviation_true"] = deviation_true
        results["big_data"]["targets_pairs"] = targets_pairs
        results["big_data"]["predictions"] = predictions
        results["big_data"]["ground_truth"] = ground_truth
        results["big_data"]["target_cell_types"] = columns

    return results
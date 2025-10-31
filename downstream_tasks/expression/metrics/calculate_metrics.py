import numpy as np
import logging


def calculate_target_genes_metrics(df_true, df_pred, target_cell_types, logger=None):
    """
    Calculates correlations for specified cell types

    Args:
        df_true (pd.DataFrame): DataFrame with true values
        df_pred (pd.DataFrame): DataFrame with predicted values
        target_cell_types (list): List of cell types to analyze

    Returns:
        dict: Dictionary with correlation metrics, np.nan for missing metrics
    """
    metrics = {}

    if logger is None:
        logger = logging.getLogger(__name__)

    # Check if all target cell types are present
    missing_in_true = [
        cell_type for cell_type in target_cell_types if cell_type not in df_true.columns
    ]
    missing_in_pred = [
        cell_type for cell_type in target_cell_types if cell_type not in df_pred.columns
    ]

    if missing_in_true or missing_in_pred:
        if missing_in_true:
            logger.debug(
                f"Following cell types were not found in true data: {missing_in_true}"
            )
        if missing_in_pred:
            logger.debug(
                f"Following cell types were not found in predicted data: {missing_in_pred}"
            )

        # Log available cell types for debugging
        logger.debug(f"Available cell types in true data: {list(df_true.columns)}")
        logger.debug(f"Available cell types in predicted data: {list(df_pred.columns)}")

        metrics["corr_selected_cell_types"] = np.nan
        metrics["corr_selected_genes"] = np.nan
        return metrics

    # Filter only target cell types
    df_true_target = df_true[target_cell_types]
    df_pred_target = df_pred[target_cell_types]

    if df_true_target.empty or df_pred_target.empty:
        logger.warning("Filtered DataFrames are empty")
        metrics["corr_selected_cell_types"] = np.nan
        metrics["corr_selected_genes"] = np.nan
        return metrics

    # Calculate gene correlations (across cell types)
    gene_correlations = []
    for gene in df_true_target.index:
        gene_true = df_true_target.loc[gene]
        gene_pred = df_pred_target.loc[gene]
        if len(gene_true) > 1 and np.std(gene_true) > 0:
            assert np.std(gene_pred) != 0, f"std(gene_pred) is 0 for gene {gene}"
            try:
                corr = np.corrcoef(gene_true, gene_pred)[0, 1]
                if not np.isnan(corr):
                    gene_correlations.append(corr)
            except:
                continue

    # Calculate cell type correlations (across genes)
    cell_correlations = []
    for cell_type in df_true_target.columns:
        cell_true = df_true_target[cell_type]
        cell_pred = df_pred_target[cell_type]
        if len(cell_true) > 1:
            try:
                assert np.std(cell_pred) != 0 and np.std(cell_true) != 0, (
                    f"std(cell_pred) or std(cell_true) is 0 for cell type {cell_type}"
                )
                corr = np.corrcoef(cell_true, cell_pred)[0, 1]
                if not np.isnan(corr):
                    cell_correlations.append(corr)
            except:
                continue

    # Set metrics, np.nan if no correlations were calculated
    metrics["corr_selected_cell_types"] = (
        float(np.mean(gene_correlations)) if gene_correlations else np.nan
    )
    metrics["corr_selected_genes"] = (
        float(np.mean(cell_correlations)) if cell_correlations else np.nan
    )

    return metrics

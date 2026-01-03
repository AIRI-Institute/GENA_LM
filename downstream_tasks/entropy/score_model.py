#!/usr/bin/env python3
"""
Score model predictions against genomic annotations.

This script computes:
1. Percentage of nucleotides with prediction=1 vs prediction=0 for each feature class
2. Baseline by randomly shuffling prediction values
3. Observed-vs-baseline ratio

All calculations are limited to accessible regions.
"""

import argparse
import numpy as np
import pandas as pd
import tqdm
from pybedtools import BedTool
import random
import sys
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score model predictions against genomic annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--annotation_beds",
        type=str,
        nargs="+",
        required=True,
        help="List of .bed files with genomic annotations (exons, introns, promoters, SINEs, etc.)"
    )
    
    parser.add_argument(
        "--prediction_bedgraphs",
        type=str,
        nargs="+",
        required=True,
        help="List of .bedGraph files with model predictions (values 0 or 1)"
    )
    
    parser.add_argument(
        "--accessible_regions",
        type=str,
        required=True,
        help=".bed file with accessible regions"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--feature_name_column",
        type=int,
        default=3,
        help="Column index (0-based) in BED file containing feature names/classes (default: 3, i.e., 4th column)"
    )
    
    parser.add_argument(
        "--n_shuffles",
        type=int,
        default=10,
        help="Number of shuffles for baseline calculation (default: 10)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def load_bedgraph_as_intervals(bedgraph_file):
    """
    Load bedGraph file and return as list of intervals with values.
    
    Returns list of tuples: (chrom, start, end, value)
    """
    intervals = []
    with open(bedgraph_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            fields = line.strip().split('\t')
            if len(fields) >= 4:
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                value = float(fields[3])
                intervals.append((chrom, start, end, value))
    return intervals


def get_feature_class(feature, name_column):
    """Extract feature class/name from BED feature."""
    if name_column < len(feature.fields):
        return feature.fields[name_column]
    else:
        raise ValueError(f"Name column {name_column} does not exist in feature {feature}")


def compute_coverage_stats(features_accessible, predictions_accessible):
    """
    Compute statistics for features overlapping with predictions within accessible regions.
    
    Returns:
        dataframe with columns: feature_name, predictions_sum, total_bp, pct_1, pct_0
    """        
    # Intersect accessible features with accessible predictions
    # Use -wa -wb to get both feature and prediction intervals
    intersections = features_accessible.intersect(predictions_accessible, wb=True)

    # Convert to DataFrame for easier manipulation
    # Intersection with -wa -wb gives: feature columns (4 cols) + prediction columns (4 cols) = 8 cols total
    # Format: feat_chrom, feat_start, feat_end, feat_name, pred_chrom, pred_start, pred_end, pred_value
    intersections_df = intersections.to_dataframe()
    
    if len(intersections_df) == 0:
        raise ValueError(f"Empty overlap")
    
    # agg_result = intersections.groupby('name').agg(
    #                     predictions_sum=('thickEnd', 'sum'),
    #                     total_bp=('end', lambda x: (x - intersections.loc[x.index, 'start']).sum())
    #     ).reset_index().rename(columns={'name': 'feature_name'})

    # Verify we have at least 8 columns (4 from features + 4 from predictions)
    if intersections_df.shape[1] != 8:
        raise ValueError(f"Intersection has {intersections_df.shape[1]} columns, expected 8")
    
    # Rename columns for clarity (assuming standard BED format: chrom, start, end, name/score)
    # Features: columns 0-3, Predictions: columns 4-7
    intersections_df.columns = list(range(intersections_df.shape[1]))
    feat_start_col = 1
    feat_end_col = 2
    feat_name_col = 3  # After merging, name is always in column 3
    pred_start_col = 5
    pred_end_col = 6
    pred_value_col = 7
    
    # Calculate overlap length for each intersection
    intersections_df['overlap_length'] = (intersections_df[feat_end_col] - intersections_df[feat_start_col])
    
    # Calculate weighted sum of predictions (pred_value * overlap_length)
    intersections_df['weighted_pred'] = intersections_df[pred_value_col] * intersections_df['overlap_length']
    
    # Aggregate by feature name
    agg_result = intersections_df.groupby(feat_name_col).agg(
        predictions_sum=('weighted_pred', 'sum'),
        total_bp=('overlap_length', 'sum')
    ).reset_index().rename(columns={feat_name_col: 'feature_name'})
    
    # Calculate percentages
    agg_result["pct_1"] = agg_result["predictions_sum"] / agg_result["total_bp"]
    agg_result["pct_0"] = 1 - agg_result["pct_1"]
    
    # Assert there are no NaN values
    assert np.sum(pd.isna(agg_result).values)==0

    return agg_result

def create_shuffled_predictions(predictions_bedtool, prediction_value_column=3, seed=None):
    """
    Create shuffled version of predictions by randomly permuting values.
    
    Returns a new BedTool with shuffled values.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    df = predictions_bedtool.to_dataframe()
    df.iloc[:,prediction_value_column] = np.random.permutation(df.iloc[:,prediction_value_column])
    shuffled_bedtool = BedTool.from_dataframe(df)
    return shuffled_bedtool


def process_annotations_and_predictions(
    annotation_bed_files,
    prediction_bedgraph_files,
    accessible_regions_file,
    feature_name_column,
    n_shuffles,
    seed
):
    """
    Main processing function.
    
    Returns pandas DataFrame with results.
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Load accessible regions
    print(f"Loading accessible regions from {accessible_regions_file}...")
    accessible_regions_bedtool = BedTool(accessible_regions_file).sort().merge() # merge to avoid overlapping regions
    
    results = []

    # concatenate all annotation files into one bedtool
    annotations_bedtool = BedTool(annotation_bed_files[0]).sort()
    if len(annotation_bed_files) > 1:
        annotations_bedtool = annotations_bedtool.cat(*annotation_bed_files[1:], postmerge=False)

    # merge features of the same type to avoid overlapping
    feature_col_name = annotations_bedtool.to_dataframe().columns.values[feature_name_column]
    annotations = annotations_bedtool.sort().to_dataframe().groupby(feature_col_name, group_keys=True).apply(
        lambda x: BedTool.from_dataframe(x).merge().to_dataframe()
	).reset_index().drop(columns=['level_1'])

    bed_format_columns = ["chrom", "start", "end", feature_col_name] # ensure that we have 4 columns
    annotations_bedtool = BedTool.from_dataframe(annotations.loc[:,bed_format_columns])
    
    # Process each prediction file
    for pred_file in prediction_bedgraph_files:
        print(f"  Processing prediction file: {pred_file}")
        
        # Load predictions
        predictions_df = pd.read_csv(pred_file, sep='\t', header=None, 
                                    dtype={0: str, 1: np.uint64, 2: np.uint64, 3: np.float64}
                                    )
        assert pd.isna(predictions_df[3]).sum() == 0
        bed_format_columns = [0, 1, 2, 3] # ensure that we have 4 columns
        
        # ensure that predictions are non-overlaping
        predictions_df = predictions_df.iloc[:,bed_format_columns].sort_values(by=[0,1,2,3]).reset_index(drop=True)
        def check_records_non_overlaping(x):
            if len(x) <= 1:
                return True
            next_start = x.iloc[1:,1]
            current_end = x.iloc[:-1,2]
            assert (next_start.values >= current_end.values).all(), f"prediction files contain overlapping intervals"
        predictions_df.groupby(0, group_keys=True).apply(check_records_non_overlaping)
        predictions_bedtool = BedTool.from_dataframe(predictions_df).sort()
        
        model_name = Path(pred_file).name.split('_')[0]
        
        # Intersect features with accessible regions first
        features_accessible = annotations_bedtool.intersect(accessible_regions_bedtool)
        
        if len(features_accessible) == 0:
            raise ValueError(f"No features found in accessible regions")
        
        # Also limit predictions to accessible regions
        predictions_accessible = predictions_bedtool.intersect(accessible_regions_bedtool)
        
        if len(predictions_accessible) == 0:
            raise ValueError(f"No predictions found in accessible regions")

        # Compute observed statistics
        # Use merged_feature_name_column (always 3 after merging) instead of original feature_name_column
        print(f"    Computing observed statistics...")
        stats_observed = compute_coverage_stats(features_accessible, predictions_accessible)
        stats_observed['model_name'] = model_name
        
        # Compute baseline with shuffled predictions
        print(f"    Computing baseline statistics with {n_shuffles} shuffles...")
        baseline_pct_1_list = []
        
        for shuffle_idx in tqdm.tqdm(range(n_shuffles), desc="Shuffling predictions"):
            shuffle_seed = seed + shuffle_idx if seed is not None else None
            shuffled_predictions = create_shuffled_predictions(predictions_accessible, prediction_value_column=3, seed=shuffle_seed)
            stats_shuffled = compute_coverage_stats(features_accessible, shuffled_predictions)
            
            # Store baseline pct_1 for each feature
            for _, row in stats_shuffled.iterrows():
                baseline_pct_1_list.append({
                    'feature_name': row['feature_name'],
                    'shuffle_idx': shuffle_idx,
                    'baseline_pct_1': row['pct_1']
                })
        
        # Calculate mean and std baseline pct_1 for each feature
        baseline_df = pd.DataFrame(baseline_pct_1_list)
        baseline_summary = baseline_df.groupby('feature_name')['baseline_pct_1'].agg(['mean', 'std']).reset_index()
        baseline_summary = baseline_summary.rename(columns={'mean': 'baseline_pct_1_mean', 'std': 'baseline_pct_1_std'})
        
        # Merge observed stats with baseline
        stats_observed = stats_observed.merge(baseline_summary, on='feature_name', how='left')
        
        assert not stats_observed['baseline_pct_1_mean'].isna().any(), "NaN found in baseline_pct_1_mean"
        assert not stats_observed['baseline_pct_1_std'].isna().any(), "NaN found in baseline_pct_1_std"
        
        # Calculate observed/baseline ratio
        stats_observed['observed_baseline_ratio'] = stats_observed['pct_1'] / (
            stats_observed['baseline_pct_1_mean']
        )
        
        # Select and rename columns for final output
        stats_observed = stats_observed[['feature_name', 'model_name', 'pct_1', 'pct_0', 
                                         'baseline_pct_1_mean', 'baseline_pct_1_std', 'observed_baseline_ratio', 
                                         'predictions_sum', 'total_bp']]
        stats_observed = stats_observed.rename(columns={'feature_name': 'feature_class'})
        
        results.append(stats_observed)
    
    # Combine all results
    if results:
        results_df = pd.concat(results, ignore_index=True)
    else:
        raise ValueError("Empty dataframe")
    
    return results_df


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input files
    for bed_file in args.annotation_beds:
        if not Path(bed_file).exists():
            print(f"Error: Annotation file not found: {bed_file}", file=sys.stderr)
            sys.exit(1)
    
    for bedgraph_file in args.prediction_bedgraphs:
        if not Path(bedgraph_file).exists():
            print(f"Error: Prediction file not found: {bedgraph_file}", file=sys.stderr)
            sys.exit(1)
    
    if not Path(args.accessible_regions).exists():
        print(f"Error: Accessible regions file not found: {args.accessible_regions}", file=sys.stderr)
        sys.exit(1)
    
    # Process data
    print("Starting processing...")
    results_df = process_annotations_and_predictions(
        args.annotation_beds,
        args.prediction_bedgraphs,
        args.accessible_regions,
        args.feature_name_column,
        args.n_shuffles,
        args.seed
    )
    
    # Save results
    print(f"Saving results to {args.output}...")
    results_df.to_csv(args.output, index=False)
    print(f"Done! Processed {len(results_df)} feature class / model combinations.")
    print(f"\nResults summary:")
    print(results_df.groupby(['feature_class', 'model_name'])[['pct_1', 'observed_baseline_ratio']].mean())


if __name__ == "__main__":
    main()
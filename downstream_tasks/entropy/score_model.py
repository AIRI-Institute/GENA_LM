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
import pybedtools
from pybedtools import BedTool
import random
from collections import defaultdict
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


def compute_coverage_stats(features, predictions_bedtool, accessible_regions_bedtool):
    """
    Compute statistics for features overlapping with predictions within accessible regions.
    
    Returns:
        dataframe with columns 
    """
    # Intersect features with accessible regions first
    features_accessible = features.intersect(accessible_regions_bedtool.merge())
    
    if len(features_accessible) == 0:
        raise ValueError(f"No features found in accessible regions for {features.filename}")
    
    # Also limit predictions to accessible regions
    predictions_accessible = predictions_bedtool.intersect(accessible_regions_bedtool.merge())
    
    if len(predictions_accessible) == 0:
        raise ValueError(f"No predictions found in accessible regions for {predictions_bedtool.filename}")
        
    # Intersect accessible features with accessible predictions
    # Use -wa -wb to get both feature and prediction intervals
    intersections = features_accessible.intersect(predictions_accessible, wb=True)

    # validate that we have exactly 8 columns in the intersection
    # chrom	start	end	name	- feature attributes
    # score	strand	thickStart	thickEnd - prediction attributes: chrom	start	end	predicted_value

    if len(intersections.fields) != 8:
        raise ValueError(f"Intersection has {len(intersections.fields)} columns, expected 8")
    
    agg_result = intersections.groupby('name').agg(
                        predictions_sum=('thickEnd', 'sum'),
                        total_bp=('end', lambda x: (x - intersections.loc[x.index, 'start']).sum())
        ).reset_index().rename(columns={'name': 'feature_name'})
    
    agg_result["pct_1"] = agg_result["predictions_sum"] / agg_result["feature_span_sum"]
    agg_result["pct_0"] = 1 - agg_result["pct_1"]

    # the resulting dataframe should have the following columns:
    # feature_name, predictions_sum, total_bp,
    # pct_1, pct_0 

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
    accessible_regions = BedTool(accessible_regions_file).merge().sort() # merge to avoid overlapping regions
    
    results = []

    # concatente all annotation files into one bedtool
    annotations = BedTool(annotation_bed_files[0]).sort()
    if len(annotation_bed_files) > 1:
        annotations = annotations.cat(*annotation_bed_files[1:], postmerge=False)

    # merge features of the same type to avoid overlapping
    annotations = annotations.sort().to_dataframe().groupby(feature_name_column).apply(
        lambda x: BedTool.from_dataframe(x).merge().to_dataframe()
	).reset_index().drop(columns=['level_1'])

    bed_format_columns = [0, 1, 2, feature_name_column] # ensure that we have 4 columns
    annotations = BedTool.from_dataframe(annotations.iloc[:,bed_format_columns])
    
    # Process each prediction file
    for pred_file in prediction_bedgraph_files:
        print(f"  Processing prediction file: {pred_file}")
        
        # Load predictions
        predictions_df = pd.read_csv(pred_file, sep='\t', header=None)
        bed_format_columns = [0, 1, 2, 3] # ensure that we have 4 columns
        predictions_df = predictions_df.iloc[:,bed_format_columns]
        predictions = BedTool.from_dataframe(predictions_df).sort()
        
        model_name = Path(pred_file).name.split('_')[0]
        
        # Compute observed statistics
        stats = compute_coverage_stats(annotations, predictions, accessible_regions)
        
        # Compute baseline with shuffled predictions
        ....

        # TODO: finish this function


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


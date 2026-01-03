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
        dict with keys: 'total_bp', 'pred_1_bp', 'pred_0_bp', 'pct_1', 'pct_0'
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
    intersections = features_accessible.intersect(predictions_accessible, wa=True, wb=True)
    
    total_bp = 0
    pred_1_bp = 0
    pred_0_bp = 0
    
    for intersection in intersections:
        # Feature coordinates (from -wa)
        feat_chrom = intersection.fields[0]
        feat_start = int(intersection.fields[1])
        feat_end = int(intersection.fields[2])
        
        # Prediction coordinates and value (from -wb)
        # bedGraph format: chrom, start, end, value
        pred_chrom = intersection.fields[3]
        pred_start = int(intersection.fields[4])
        pred_end = int(intersection.fields[5])
        pred_value = float(intersection.fields[6])
        
        # Only process if chromosomes match
        if feat_chrom != pred_chrom:
            continue
        
        # Calculate overlap between feature and prediction interval
        overlap_start = max(feat_start, pred_start)
        overlap_end = min(feat_end, pred_end)
        overlap_bp = max(0, overlap_end - overlap_start)
        
        if overlap_bp > 0:
            total_bp += overlap_bp
            # Check if prediction value is 1 or 0 (allowing for floating point comparison)
            if abs(pred_value - 1.0) < 0.001:
                pred_1_bp += overlap_bp
            elif abs(pred_value - 0.0) < 0.001:
                pred_0_bp += overlap_bp
            # If value is neither 0 nor 1, we still count it in total_bp but not in pred_1_bp or pred_0_bp
    
    pct_1 = (pred_1_bp / total_bp * 100.0) if total_bp > 0 else 0.0
    pct_0 = (pred_0_bp / total_bp * 100.0) if total_bp > 0 else 0.0
    
    return {
        'total_bp': total_bp,
        'pred_1_bp': pred_1_bp,
        'pred_0_bp': pred_0_bp,
        'pct_1': pct_1,
        'pct_0': pct_0
    }


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
    annotations = BedTool.from_dataframe(annotations)
    
    # Process each prediction file
    for pred_file in prediction_bedgraph_files:
        print(f"  Processing prediction file: {pred_file}")
        
        # Load predictions
        predictions = BedTool(pred_file)
        model_name = Path(pred_file).stem
        
        # Group features by class
        feature_classes = defaultdict(list)
        for feature in annotations:
            feature_class = get_feature_class(feature, feature_name_column)
            feature_classes[feature_class].append(feature)
        
        # Process each feature class
        for feature_class, features_list in feature_classes.items():
            print(f"    Processing feature class: {feature_class} ({len(features_list)} features)")
            
            # Create BedTool for this feature class
            # Preserve all BED fields
            bed_lines = []
            for f in features_list:
                line = f"{f.chrom}\t{f.start}\t{f.end}"
                # Add remaining fields if they exist
                if len(f.fields) > 3:
                    line += "\t" + "\t".join(f.fields[3:])
                bed_lines.append(line)
            features_bedtool = BedTool("\n".join(bed_lines), from_string=True)
            
            # Compute observed statistics
            stats = compute_coverage_stats(features_bedtool, predictions, accessible_regions)
            
            # Compute baseline with shuffled predictions
            baseline_pct_1_list = []
            for shuffle_idx in range(n_shuffles):
                shuffled_preds = create_shuffled_predictions(predictions, seed=seed + shuffle_idx)
                baseline_stats = compute_coverage_stats(features_bedtool, shuffled_preds, accessible_regions)
                baseline_pct_1_list.append(baseline_stats['pct_1'])
            
            baseline_pct_1_mean = np.mean(baseline_pct_1_list) if baseline_pct_1_list else 0.0
            baseline_pct_1_std = np.std(baseline_pct_1_list) if baseline_pct_1_list else 0.0
            
            # Compute observed-vs-baseline ratio
            # Handle division by zero: if baseline is 0 and observed is 0, ratio is 1.0 (no enrichment)
            # If baseline is 0 but observed > 0, set ratio to a large value (or NaN)
            if baseline_pct_1_mean > 0:
                observed_baseline_ratio = stats['pct_1'] / baseline_pct_1_mean
            elif stats['pct_1'] == 0:
                observed_baseline_ratio = 1.0  # Both are 0, no enrichment
            else:
                observed_baseline_ratio = np.nan  # Observed > 0 but baseline = 0 (infinite enrichment)
            
            # Store results
            results.append({
                'annotation_file': Path(ann_file).name,
                'model_name': model_name,
                'feature_class': feature_class,
                'total_bp': stats['total_bp'],
                'pred_1_bp': stats['pred_1_bp'],
                'pred_0_bp': stats['pred_0_bp'],
                'pct_1': stats['pct_1'],
                'pct_0': stats['pct_0'],
                'baseline_pct_1_mean': baseline_pct_1_mean,
                'baseline_pct_1_std': baseline_pct_1_std,
                'observed_baseline_ratio': observed_baseline_ratio,
                'n_shuffles': n_shuffles
            })

    return pd.DataFrame(results)


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


#!/usr/bin/env python3
"""
Script to process bigWig files and compute TSS/PolyA metrics.
Reads bigWig files, applies threshold, combines strands, and generates overlap plots.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import pyBigWig as bw

# Import the metric computation functions
sys.path.append('../annotation/compute_metric')
from tss_polya_metric_v2 import compute_metrics


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process bigWig files and compute TSS/PolyA metrics')
    parser.add_argument('--bigwig_path', type=str, required=True,
                       help='Path to directory containing bigWig files')
    parser.add_argument('--gt_path', type=str,
						default='../annotation/compute_metric/NC_060944.1.tsv',
                       help='Path to ground truth TSV file')
    parser.add_argument('--threshold', type=float,
						default=0.5,
                       help='Threshold for converting bigWig values to binary predictions')
    parser.add_argument('--max_k', type=int,
						default=50,
                       help='Maximum number of peaks to consider for overlap calculation')
    return parser.parse_args()


def validate_bigwig_files(bigwig_path):
    """Validate that all 4 required bigWig files exist."""
    required_files = []
    for class_name in ["tss", "polya"]:
        for strand in ["+", "-"]:
            filename = f"{class_name}_{strand}.bw"
            required_files.append(filename)
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(bigwig_path, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        raise FileNotFoundError(f"Missing bigWig files: {missing_files}")
    
    print(f"✓ Found all required bigWig files in {bigwig_path}")
    return required_files


def read_bigwig_chromosome(bigwig_path, filename, chromosome="NC_060944.1"):
    """Read all values from a specific chromosome in a bigWig file."""
    filepath = os.path.join(bigwig_path, filename)
    
    with bw.open(filepath) as bw_file:
        # Get chromosome length
        chrom_length = bw_file.chroms()[chromosome]
        print(f"Reading {filename}: chromosome {chromosome} (length: {chrom_length})")
        
        # Read all values for the chromosome
        values = bw_file.values(chromosome, 0, chrom_length)
        
        # Convert None values to 0 (pyBigWig returns None for missing values)
        values = [0.0 if v is None else v for v in values]
        
    return np.array(values, dtype=np.float32)


def apply_threshold(values, threshold):
    """Apply threshold to convert values to binary 0/1 predictions."""
    return (values >= threshold).astype(int)


def combine_strands(tss_plus, tss_minus, polya_plus, polya_minus):
    """Combine strand predictions using max() operation."""
    tss_combined = np.maximum(tss_plus, tss_minus)
    polya_combined = np.maximum(polya_plus, polya_minus)
    return tss_combined, polya_combined


def compute_metrics_and_plot(predictions, gt_path, label, output_path, max_k=50):
    """Compute overlap metrics and generate plot."""
    print(f"Computing metrics for {label}...")
    
    # Load ground truth data
    df = pd.read_csv(gt_path, sep='\t')

    compute_metrics(predictions, df=df, label=label, output_path=output_path, max_k=max_k)
    


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.bigwig_path):
        raise FileNotFoundError(f"BigWig path does not exist: {args.bigwig_path}")
    
    if not os.path.exists(args.gt_path):
        raise FileNotFoundError(f"GT file does not exist: {args.gt_path}")
    
    # Validate bigWig files
    _ = validate_bigwig_files(args.bigwig_path)
    
    # Set output directory (same as bigWig directory)
    output_dir = args.bigwig_path
    
    print(f"Processing with threshold: {args.threshold}")
    print(f"Output directory: {output_dir}")
    
    # Read bigWig files for chromosome NC_060945.1
    print("\nReading bigWig files...")
    tss_plus = read_bigwig_chromosome(args.bigwig_path, "tss_+.bw")
    tss_minus = read_bigwig_chromosome(args.bigwig_path, "tss_-.bw")
    polya_plus = read_bigwig_chromosome(args.bigwig_path, "polya_+.bw")
    polya_minus = read_bigwig_chromosome(args.bigwig_path, "polya_-.bw")
    
    # Apply threshold to convert to binary predictions
    print(f"\nApplying threshold {args.threshold}...")
    tss_plus_binary = apply_threshold(tss_plus, args.threshold)
    tss_minus_binary = apply_threshold(tss_minus, args.threshold)
    polya_plus_binary = apply_threshold(polya_plus, args.threshold)
    polya_minus_binary = apply_threshold(polya_minus, args.threshold)
    
    # Combine strands using max()
    print("Combining strands...")
    tss_combined, polya_combined = combine_strands(
        tss_plus_binary, tss_minus_binary, 
        polya_plus_binary, polya_minus_binary
    )
    
    print(f"TSS combined: {np.sum(tss_combined)} positive predictions out of {len(tss_combined)}")
    print(f"PolyA combined: {np.sum(polya_combined)} positive predictions out of {len(polya_combined)}")
    
    # Generate output filenames with threshold
    tss_output = os.path.join(output_dir, f"TSS_threshold_{args.threshold}_max_k_{args.max_k}.png")
    polya_output = os.path.join(output_dir, f"PolyA_threshold_{args.threshold}_max_k_{args.max_k}.png")
    
    # Compute metrics and generate plots
    print("\nComputing TSS metrics...")
    _ = compute_metrics_and_plot(
        tss_combined, args.gt_path, "TSS", tss_output,
        max_k=args.max_k
    )
    
    print("\nComputing PolyA metrics...")
    _ = compute_metrics_and_plot(
        polya_combined, args.gt_path, "PolyA", polya_output,
        max_k=args.max_k
    )
    
    print(f"\n✓ Processing complete!")
    print(f"✓ TSS plot saved to: {tss_output}")
    print(f"✓ PolyA plot saved to: {polya_output}")


if __name__ == "__main__":
    main()

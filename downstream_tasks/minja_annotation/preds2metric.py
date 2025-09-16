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
    parser.add_argument('--postprocess', action='store_true', default=False,
                       help='Apply postprocessing to keep only highest signal bp in continuous regions')
    parser.add_argument('--min_dist', type=int, default=0,
                       help='Minimum distance: fill gaps shorter than min_dist with 1s before postprocessing')
    parser.add_argument('--strand', type=str, default="+",
                       help='Strand to process')
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

def merge_rev_comp(signal_plus, signal_minus):
    """Merge plus and minus strands of a signal."""
    return np.mean([signal_plus, signal_minus], axis=0)


def postprocess_signal(binary_signal, original_signal, min_dist=0):
    """
    Postprocess binary signal to keep only the highest signal bp in continuous regions.
    First fills gaps shorter than min_dist with 1s, then applies peak-finding logic.
    Uses vectorized operations for faster processing.
    
    Args:
        binary_signal: Binary array (0/1) indicating signal presence
        original_signal: Original signal values used to determine highest points
        min_dist: Minimum distance - gaps shorter than this are filled with 1s
    
    Returns:
        Postprocessed binary signal with only highest signal bp in each continuous region
    """
    if len(binary_signal) != len(original_signal):
        raise ValueError("Binary signal and original signal must have the same length")
    
    # Convert to numpy arrays for vectorized operations
    binary = np.array(binary_signal, dtype=int)
    original = np.array(original_signal, dtype=float)
    
    # Step 1: Fill gaps shorter than min_dist with 1s
    if min_dist > 0:
        binary = fill_short_gaps(binary, min_dist)
    
    # Step 2: Find region starts and ends using vectorized operations
    # Region starts: where binary goes from 0 to 1
    shifted_binary = np.concatenate([[0], binary[:-1]])  # Shift right by 1
    region_starts = np.where((binary == 1) & (shifted_binary == 0))[0]
    
    # Region ends: where binary goes from 1 to 0
    shifted_binary_end = np.concatenate([binary[1:], [0]])  # Shift left by 1
    region_ends = np.where((binary == 1) & (shifted_binary_end == 0))[0]
    
    # Handle case where region extends to the end of the array
    if len(region_starts) > len(region_ends):
        region_ends = np.concatenate([region_ends-1, [len(binary)]])
    
    # Initialize output array
    postprocessed = np.zeros_like(binary)

    print ("Mean positive region length: ",np.mean(region_ends - region_starts))
    
    # Step 3: Process each region - keep only highest signal bp
    for start, end in zip(region_starts, region_ends):
        # Find the position with highest original signal in this region
        # Add +1 to end to include the last "1" in the region
        region_original = original[start:end+1]
        max_idx = np.argmax(region_original)
        max_position = start + max_idx
        postprocessed[max_position] = 1
    
    print ("Sum of postprocessed: ",np.sum(postprocessed), " sum of the original: ",np.sum(binary))
    return postprocessed


def fill_short_gaps(binary_signal, min_dist):
    """
    Fill gaps (sequences of 0s) shorter than min_dist with 1s.
    
    Args:
        binary_signal: Binary array (0/1)
        min_dist: Minimum distance - gaps shorter than this are filled
    
    Returns:
        Binary array with short gaps filled
    """
    if min_dist <= 0:
        return binary_signal
    
    filled = binary_signal.copy()
    
    # Find gap starts and ends using vectorized operations
    # Gap starts: where signal goes from 1 to 0
    shifted_binary = np.concatenate([[1], binary_signal[:-1]])  # Shift right by 1
    gap_starts = np.where((binary_signal == 0) & (shifted_binary == 1))[0]
    
    # Gap ends: where signal goes from 0 to 1
    shifted_binary_end = np.concatenate([binary_signal[1:], [1]])  # Shift left by 1
    gap_ends = np.where((binary_signal == 0) & (shifted_binary_end == 1))[0]
    
    # Handle case where gap extends to the end of the array
    if len(gap_starts) > len(gap_ends):
        gap_ends = np.concatenate([gap_ends, [len(binary_signal)]])
    
    # Fill gaps shorter than min_dist
    for start, end in zip(gap_starts, gap_ends):
        gap_length = end - start
        if gap_length <= min_dist:
            filled[start:end+1] = 1
    
    return filled

def compute_metrics_and_plot(predictions, gt_path, label, output_path, max_k=250):
    """Compute overlap metrics and generate plot."""
    print(f"Computing metrics for {label}...")
    
    # Load ground truth data
    df = pd.read_csv(gt_path, sep='\t')

    results = compute_metrics(predictions, df=df, label=label, output_path=output_path, max_k=max_k)
    return results

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
    print(f"Postprocessing enabled: {args.postprocess}")
    if args.postprocess:
        print(f"Minimum distance for gap filling: {args.min_dist}")
    print(f"Output directory: {output_dir}")
    
    # Read bigWig files for chromosome NC_060945.1
    print("\nReading bigWig files...")

    if args.strand == "+":
        tss_plus = read_bigwig_chromosome(args.bigwig_path, "tss_+.bw")
        tss_minus = read_bigwig_chromosome(args.bigwig_path, "tss_-.bw")
        polya_plus = read_bigwig_chromosome(args.bigwig_path, "polya_+.bw")
        polya_minus = read_bigwig_chromosome(args.bigwig_path, "polya_-.bw")
    elif args.strand == "-":
        tss_plus = read_bigwig_chromosome(args.bigwig_path, "tss_-rev_comp_.bw")
        tss_minus = read_bigwig_chromosome(args.bigwig_path, "tss_+rev_comp_.bw")
        polya_plus = read_bigwig_chromosome(args.bigwig_path, "polya_-rev_comp_.bw")
        polya_minus = read_bigwig_chromosome(args.bigwig_path, "polya_+rev_comp_.bw")
    elif args.strand == "both":
        tss_plus = merge_rev_comp(read_bigwig_chromosome(args.bigwig_path, "tss_+.bw"), read_bigwig_chromosome(args.bigwig_path, "tss_-rev_comp_.bw"))
        tss_minus = merge_rev_comp(read_bigwig_chromosome(args.bigwig_path, "tss_-.bw"), read_bigwig_chromosome(args.bigwig_path, "tss_+rev_comp_.bw"))
        polya_plus = merge_rev_comp(read_bigwig_chromosome(args.bigwig_path, "polya_+.bw"), read_bigwig_chromosome(args.bigwig_path, "polya_-rev_comp_.bw"))
        polya_minus = merge_rev_comp(read_bigwig_chromosome(args.bigwig_path, "polya_-.bw"), read_bigwig_chromosome(args.bigwig_path, "polya_+rev_comp_.bw"))
    else:
        raise ValueError(f"Invalid strand: {args.strand}")
    
    # Apply threshold to convert to binary predictions
    print(f"\nApplying threshold {args.threshold}...")
    tss_plus_binary = apply_threshold(tss_plus, args.threshold)
    tss_minus_binary = apply_threshold(tss_minus, args.threshold)
    polya_plus_binary = apply_threshold(polya_plus, args.threshold)
    polya_minus_binary = apply_threshold(polya_minus, args.threshold)
    
    # Apply postprocessing if requested
    if args.postprocess:
        print(f"Applying postprocessing to keep only highest signal bp in continuous regions...")
        if args.min_dist > 0:
            print(f"Filling gaps shorter than {args.min_dist} bp with 1s...")
        tss_plus_binary = postprocess_signal(tss_plus_binary, tss_plus, args.min_dist)
        tss_minus_binary = postprocess_signal(tss_minus_binary, tss_minus, args.min_dist)
        polya_plus_binary = postprocess_signal(polya_plus_binary, polya_plus, args.min_dist)
        polya_minus_binary = postprocess_signal(polya_minus_binary, polya_minus, args.min_dist)
    
    # Combine strands using max()
    print("Combining strands...")
    tss_combined, polya_combined = combine_strands(
        tss_plus_binary, tss_minus_binary, 
        polya_plus_binary, polya_minus_binary
    )
    
    print(f"TSS combined: {np.sum(tss_combined)} positive predictions out of {len(tss_combined)}")
    print(f"PolyA combined: {np.sum(polya_combined)} positive predictions out of {len(polya_combined)}")
    
    # Generate output filenames with threshold and postprocessing flags
    postprocess_suffix = "_postprocessed" if args.postprocess else ""
    min_dist_suffix = f"_mindist_{args.min_dist}" if args.postprocess and args.min_dist > 0 else ""
    strand_suffix = f"_strand_{args.strand}" if args.strand != "+" else ""
    tss_output = os.path.join(output_dir, f"TSS_threshold_{args.threshold}_max_k_{args.max_k}{postprocess_suffix}{min_dist_suffix}{strand_suffix}.png")
    polya_output = os.path.join(output_dir, f"PolyA_threshold_{args.threshold}_max_k_{args.max_k}{postprocess_suffix}{min_dist_suffix}{strand_suffix}.png")
    
    # Compute metrics and generate plots
    print("\nComputing TSS metrics...")
    results_tss = compute_metrics_and_plot(
        tss_combined, args.gt_path, "TSS", tss_output,
        max_k=args.max_k
    )
    
    print("\nComputing PolyA metrics...")
    results_polya = compute_metrics_and_plot(
        polya_combined, args.gt_path, "PolyA", polya_output,
        max_k=args.max_k
    )

    # finally, write the metrics to a text file
    with open(
        os.path.join(output_dir, f"metrics_threshold_{args.threshold}_max_k_{args.max_k}{postprocess_suffix}{min_dist_suffix}{strand_suffix}.txt"),
        "w"
    ) as f:
        # TSS O (k=50)	TSS NO (k=50)	PolyA O (k=50)	PolyA NO (k=50)	TSS O (k=250)	TSS NO (k=250)	PolyA O (k=250)	PolyA NO (k=250)
        for k in [50, 250]:
            for r in [results_tss, results_polya]:
                for t in ["O", "NO"]:
                    f.write(str(r[t + "_" + str(k)]) + "\t")
        f.write("\n")
    
    print(f"\n✓ Processing complete!")
    print(f"✓ TSS plot saved to: {tss_output}")
    print(f"✓ PolyA plot saved to: {polya_output}")


if __name__ == "__main__":
    main()

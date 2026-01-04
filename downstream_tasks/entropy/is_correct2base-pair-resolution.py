#!/usr/bin/env python3
"""
Convert is_correct bedgraph files (token-level) to base-pair resolution.

This script:
1. Reads _highest_prob_token.bed files containing predicted tokens
2. Fetches true sequences from the genome
3. Compares predicted vs true sequences base-by-base
4. Writes is_correct-bp-res.bedgraph files with correctness per base pair
5. Uses is_correct.bedgraph for sanity checks
"""

import argparse
import pysam
import sys
import gzip
from pathlib import Path
from tqdm import tqdm


def parse_bed_line(line):
    """Parse a BED line and return (chrom, start, end, value/sequence)."""
    fields = line.strip().split('\t')
    if len(fields) != 4:
        raise ValueError(f"Wrong format for line {line}")
    chrom = fields[0]
    start = int(fields[1])
    end = int(fields[2])
    value = fields[3]
    return chrom, start, end, value


def compare_sequences(predicted_seq, true_seq):
    """
    Compare two sequences base-by-base.
    Returns a list of 1s (correct) and 0s (incorrect) for each position.
    If predicted is longer, truncate it; if it is shorter, pad with an arbitrary symbol ('N').
    """
    len_pred = len(predicted_seq)
    len_true = len(true_seq)

    # Pad or truncate predicted_seq to match true_seq length
    if len_pred > len_true:
        predicted_seq = predicted_seq[:len_true]
    elif len_pred < len_true:
        # Repeat the predicted sequence as many times as needed to match true_seq length
        n_repeats = (len_true + len_pred - 1) // len_pred  # ceil division
        predicted_seq = (predicted_seq * n_repeats)[:len_true]

    # Now both sequences should have the same length
    correctness = []
    for pred_base, true_base in zip(predicted_seq, true_seq):
        correctness.append(1 if pred_base.upper() == true_base.upper() else 0)
    
    return correctness


def process_files(highest_prob_token_file, is_correct_file, genome_path, output_file):
    """
    Process files line by line and generate base-pair resolution correctness.
    
    Args:
        highest_prob_token_file: Path to _highest_prob_token.bed file
        is_correct_file: Path to is_correct.bedgraph file
        genome_path: Path to genome FASTA file
        output_file: Path to output is_correct-bp-res.bedgraph file
    """
    # Open genome FASTA file
    try:
        fasta = pysam.FastaFile(genome_path)
    except Exception as e:
        print(f"Error opening genome file {genome_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Count total lines for progress bar
    print("Counting lines in input files...", file=sys.stderr)
    with open(highest_prob_token_file, 'r') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    # Open input files
    try:
        with open(highest_prob_token_file, 'r') as f_pred, \
             open(is_correct_file, 'r') as f_correct:
            
            # Open output file (gzipped)
            with gzip.open(output_file, 'wt') as f_out:
                line_num = 0
                
                # Read files line by line simultaneously with progress bar
                for pred_line, correct_line in tqdm(zip(f_pred, f_correct), 
                                                    total=total_lines, 
                                                    desc="Processing tokens",
                                                    unit="tokens"):
                    line_num += 1
                    
                    # Skip empty lines
                    if not pred_line.strip() or not correct_line.strip():
                        continue
                    
                    # Parse lines
                    pred_data = parse_bed_line(pred_line)
                    correct_data = parse_bed_line(correct_line)
                    
                    chrom_pred, start_pred, end_pred, predicted_seq = pred_data
                    chrom_correct, start_correct, end_correct, is_correct_value = correct_data
                    
                    # Verify coordinates match
                    if chrom_pred != chrom_correct or start_pred != start_correct or end_pred != end_correct:
                        raise ValueError(f"Coordinate mismatch at line {line_num}: "
                              f"predicted={chrom_pred}:{start_pred}-{end_pred}, "
                              f"is_correct={chrom_correct}:{start_correct}-{end_correct}")
                    
                    # Fetch true sequence from genome
                    try:
                        true_seq = fasta.fetch(chrom_pred, start_pred, end_pred).upper()
                    except Exception as e:
                        raise ValueError(f"Error fetching sequence at line {line_num} "
                              f"({chrom_pred}:{start_pred}-{end_pred}): {e}")
                    
                    # Sanity check: if is_correct is 1, predicted sequence should match genome exactly
                    if is_correct_value == '1':
                        if predicted_seq.upper() != true_seq:
                            raise ValueError(f"Sanity check failed at line {line_num}: "
                                  f"is_correct=1 but predicted_seq != true_seq\n"
                                  f"  predicted: {predicted_seq}\n"
                                  f"  true:      {true_seq}")
                    
                    # Compare sequences base-by-base
                    correctness = compare_sequences(predicted_seq, true_seq)
                    
                    # Write output: one line per base pair
                    for i, correct in enumerate(correctness):
                        bp_start = start_pred + i
                        bp_end = bp_start + 1
                        f_out.write(f"{chrom_pred}\t{bp_start}\t{bp_end}\t{correct}\n")
                
    except Exception as e:
        raise ValueError(f"Error processing files: {e}")
    finally:
        fasta.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert token-level is_correct bedgraph to base-pair resolution"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix for input and output files. Files should be named {prefix}_highest_prob_token.bed, {prefix}_is_correct.bedgraph, {prefix}_is_correct_bp.bedgraph.gz"
    )
    parser.add_argument(
        "--genome",
        type=str,
        required=True,
        help="Path to genome FASTA file"
    )
    
    args = parser.parse_args()
    
    # Construct file paths from prefix
    highest_prob_token_file = f"{args.prefix}_highest_prob_token.bed"
    is_correct_file = f"{args.prefix}_is_correct.bedgraph"
    output_file = f"{args.prefix}_is_correct_bp.bedgraph.gz"
    
    # Validate input files exist
    for file_path, name in [(highest_prob_token_file, "highest_prob_token"),
                            (is_correct_file, "is_correct"),
                            (args.genome, "genome")]:
        if not Path(file_path).exists():
            print(f"Error: {name} file not found: {file_path}", file=sys.stderr)
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing files...")
    print(f"  Predicted tokens: {highest_prob_token_file}")
    print(f"  Is correct: {is_correct_file}")
    print(f"  Genome: {args.genome}")
    print(f"  Output: {output_file}")
    
    process_files(highest_prob_token_file, is_correct_file, args.genome, output_file)
    
    print(f"Done! Output written to {output_file}")


if __name__ == "__main__":
    main()


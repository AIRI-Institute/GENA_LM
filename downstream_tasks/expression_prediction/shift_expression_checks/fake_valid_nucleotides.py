# python3 downstream_tasks/expression_prediction/shift_expression_checks/fake_valid_nucleotides.py --fasta downstream_tasks/expression_prediction/datasets/data/genomes/hg38/hg38.fa --bed downstream_tasks/expression_prediction/intervals/borzoi_human_valid.bed --output downstream_tasks/expression_prediction/datasets/data/genomes/hg38/hg38.faked.fa

import pysam
import random
import argparse
from collections import defaultdict
import pybedtools
from tqdm import tqdm

def merge_overlapping_intervals(bed_file):
    """Merge overlapping intervals from a BED file."""
    # Use pybedtools to merge overlapping intervals
    bed = pybedtools.BedTool(bed_file).sort()
    merged = []
    
    # Merge overlapping intervals and convert to tuples
    for interval in bed.merge():
        merged.append((interval.chrom, interval.start, interval.end))
        
    return merged

def substitute_nucleotides(fasta_file, merged_intervals, output_file):
    """Substitute nucleotides within intervals with random nucleotides."""
    nucleotides = ['A', 'T', 'G', 'C']
    
    # Create a dictionary of intervals for faster lookup
    interval_dict = defaultdict(list)
    for chrom, start, end in merged_intervals:
        interval_dict[chrom].append((start, end))
    
    # Calculate total number of nucleotides to be substituted
    total_substitutions = sum(end - start for intervals in interval_dict.values() for start, end in intervals)
    
    # Create progress bar
    pbar = tqdm(total=total_substitutions, desc="Substituting nucleotides")
    
    # Open input and output FASTA files
    with pysam.FastxFile(fasta_file) as in_fasta, open(output_file, 'w') as out_fasta:
        for record in in_fasta:
            sequence = list(record.sequence)
            chrom = record.name
            
            # Check if chromosome has any intervals
            if chrom in interval_dict:
                for start, end in interval_dict[chrom]:
                    # Ensure we don't go beyond sequence length
                    start = max(0, start)
                    end = min(len(sequence), end)
                    
                    # Substitute nucleotides in the interval
                    for i in range(start, end):
                        sequence[i] = random.choice(nucleotides)
                        pbar.update(1)
            
            # Write modified sequence
            out_fasta.write(f">{record.name}\n")
            out_fasta.write(''.join(sequence) + '\n')
    
    pbar.close()

def main():
    parser = argparse.ArgumentParser(description='Process genome sequences and intervals')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--bed', required=True, help='Input BED file')
    parser.add_argument('--output', required=True, help='Output FASTA file')
    
    args = parser.parse_args()
    
    # Merge overlapping intervals
    merged_intervals = merge_overlapping_intervals(args.bed)
    
    # Process FASTA file and substitute nucleotides
    substitute_nucleotides(args.fasta, merged_intervals, args.output)

if __name__ == '__main__':
    main() 
# python3 process_h5_predictions.py --h5_path data/decoder-models/minja_human_chr21_only_hg38_evo2_mlm_embeddings.h5 --genome_path ../expression_prediction/datasets/data/genomes/hg38/hg38.fa --output_prefix data/decoder-models/evo2_chr21 --token_map evo2
# python3 process_h5_predictions.py --h5_path data/decoder-models/minja_human_chr21_only_hg38_caduceus_ps_mlm_embeddings.h5 --genome_path ../expression_prediction/datasets/data/genomes/hg38/hg38.fa --output_prefix data/decoder-models/caduceus_ps_chr21 --token_map caduceus
# python3 process_h5_predictions.py --h5_path data/decoder-models/minja_human_chr21_only_hg38_caduceus_ph_mlm_embeddings.h5 --genome_path ../expression_prediction/datasets/data/genomes/hg38/hg38.fa --output_prefix data/decoder-models/caduceus_ph_chr21 --token_map caduceus

"""
Process pre-computed model predictions from h5 file and generate metrics files.

This script reads logits from an h5 file and generates the same output files
as compute_entropy.py, but without needing to run the model.
"""

import os
import numpy as np
import pysam
import h5py
import torch
from scipy.stats import entropy
import argparse
import datetime
from tqdm import tqdm
from pathlib import Path
import datetime
from torch.distributions import Categorical

def token_id_to_base(token_id, token_map):
    """
    Convert token ID to base character.
    
    Args:
        token_id: Token ID
        token_map: Mapping from token_id -> base character
    
    Returns:
        str: Base character ('A', 'C', 'G', 'T', or 'N' if not in map)
    """
    return token_map.get(token_id, 'N')


def parse_h5_key(key):
    """
    Parse h5 key to extract chromosome and coordinates.
    
    Key format: '>chr:start-end,start-end'
    The first pair is original coordinates (with chromosome), second pair is hg38 coordinates (same chromosome, no chr prefix).
    
    Returns:
        tuple: (chrom, hg38_start, hg38_end, orig_start, orig_end)
    """
    if not key.startswith('>'):
        raise ValueError(f"Invalid key format (must start with '>'): {key}")
    
    key = key[1:]  # Remove '>'
    parts = key.split(',')
    if len(parts) != 2:
        raise ValueError(f"Invalid key format (must contain exactly one comma): {key}")
    
    # Parse original coordinates (includes chromosome name)
    orig_part = parts[0]
    if ':' not in orig_part or '-' not in orig_part:
        raise ValueError(f"Invalid coordinate format: {orig_part}")
    orig_chrom, orig_coords = orig_part.split(':')
    orig_start, orig_end = map(int, orig_coords.split('-'))
    
    # Parse hg38 coordinates (no chromosome name, uses same chromosome as first part)
    hg38_part = parts[1]
    if '-' not in hg38_part:
        raise ValueError(f"Invalid hg38 coordinate format: {hg38_part}")
    hg38_start, hg38_end = map(int, hg38_part.split('-'))
    
    # Use chromosome from original coordinates for hg38
    hg38_chrom = orig_chrom
    
    return hg38_chrom, hg38_start, hg38_end, orig_start, orig_end


def compute_metrics_from_logits(logits, token_map, ground_truth_bases):
    """
    Compute metrics from logits.
    
    Args:
        logits: Array of shape (n_positions, vocab_size)
        token_map: Mapping from token_id -> base character
        ground_truth_bases: Array of ground truth base characters (length n_positions)
    
    Returns:
        dict: Metrics for each position
    """
    # Convert logits to probabilities
    
    logits_tensor = torch.from_numpy(logits).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).detach()
    probs = torch.softmax(logits_tensor, dim=1).cpu().numpy()
    
    # Get highest probability token IDs
    highest_prob_token_ids = np.argmax(probs, axis=1)
    highest_probs = np.max(probs, axis=1)
    
    # Convert token IDs to bases
    predicted_bases = np.array([token_id_to_base(tid, token_map) for tid in highest_prob_token_ids])
    
    # Check correctness (only for valid bases: A, C, G, T)
    is_correct = (predicted_bases == ground_truth_bases).astype(int)
    
    # Calculate entropy for each position
    distr = Categorical(logits=logits_tensor)
    entropies = distr.entropy().cpu().numpy() / np.log(2)       
    
    return {
        "is_correct": is_correct,
        "entropy": entropies,
        "highest_prob": highest_probs,
        "highest_prob_token": predicted_bases
    }


def process_h5_file(h5_path, genome_path, token_map, output_prefix):
    """
    Process h5 file and generate output files.
    
    Args:
        h5_path: Path to h5 file with logits
        genome_path: Path to hg38 genome FASTA file
        token_map: Mapping from token_id -> base character
        output_prefix: Prefix for output files
    """
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open genome file
    fasta = pysam.FastaFile(genome_path)
    
    # Open output files
    metric_types = {
        "is_correct": "bedgraph",
        "entropy": "bedgraph",
        "highest_prob": "bedgraph",
        "highest_prob_token": "bed"
    }
    file_handlers = {
        metric: open(f"{output_prefix}_{metric}.{ftype}", 'w')
        for metric, ftype in metric_types.items()
    }
    
    # Write run parameters
    with open(f"{output_prefix}_run_params.txt", "w") as fout:
        fout.write(f"date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fout.write(f"h5_path: {h5_path}\n")
        fout.write(f"genome_path: {genome_path}\n")
        fout.write(f"token_map: {token_map}\n")
        fout.write(f"output_prefix: {output_prefix}\n")
    
    # Track processed positions to avoid duplicates
    processed_positions = set()
    
    # Open h5 file and process each key
    with h5py.File(h5_path, 'r') as h5_file:
        keys = list(h5_file.keys())
        
        for key in tqdm(keys, desc="Processing regions"):
            try:
                # Parse coordinates
                chrom, hg38_start, hg38_end, orig_start, orig_end = parse_h5_key(key)
                
                # Get logits
                logits = np.array(h5_file[key]['logits'])
                
                # Remove batch dimension if present
                if logits.ndim == 3 and logits.shape[0] == 1:
                    logits = logits[0]  # Shape: (n_positions, vocab_size)
                elif logits.ndim == 3:
                    raise ValueError(f"Logits have unexpected dimensions: {logits.ndim}. Expected 2 or 3, and for ndim=3, batch size should be 1.")
                elif logits.ndim != 2:
                    raise ValueError(f"Logits have unexpected dimensions: {logits.ndim}. Expected 2 or 3, and for ndim=3, batch size should be 1.")
                
                n_positions = logits.shape[0]
                seq_length = hg38_end - hg38_start
                
                # Check if sequence length matches number of positions
                if seq_length != n_positions:
                    raise ValueError(f"Sequence length ({seq_length}) != number of positions ({n_positions}) for {key}. Skipping.")
                
                # Note: how inference was done:
                # we process chunks of 32KB shifting by 16KB, assuming that 16 KB is minimal reasonalbe context for the model
                # also we make prediction for each base, we will now only use the predictions from position 16001 onwards
                # Skip chunks with length <= 16KB
                if seq_length <= 16000:
                    continue
                
                # For chunks > 16KB, process only bases from position 16001 (index 16000) onwards
                skip_bases = 16000
                start_idx = skip_bases
                
                # Get ground truth sequence from genome
                try:
                    ground_truth_seq = fasta.fetch(chrom, hg38_start, hg38_end).upper()
                except Exception as e:
                    raise ValueError(f"Error: Could not fetch sequence for {chrom}:{hg38_start}-{hg38_end}: {e}. Skipping.")
                
                if len(ground_truth_seq) != n_positions:
                    raise ValueError(f"Ground truth length ({len(ground_truth_seq)}) != number of positions ({n_positions}) for {key}. Skipping.")
                
                # Slice arrays to start from position 16001
                logits = logits[start_idx:]
                ground_truth_seq = ground_truth_seq[start_idx:]
                ground_truth_bases = np.array(list(ground_truth_seq))
                
                # Update starting position for output coordinates
                adjusted_start = hg38_start + start_idx
                n_positions_to_process = len(ground_truth_bases)
                
                # Compute metrics
                metrics = compute_metrics_from_logits(logits, token_map, ground_truth_bases)
                # Write results (one line per position), skipping duplicates
                for i in range(n_positions_to_process):
                    pos_start = adjusted_start + i
                    pos_end = pos_start + 1
                    
                    # Skip if this position has already been processed
                    position_key = (chrom, pos_start)
                    if position_key in processed_positions:
                        raise ValueError(f"Position {position_key} has already been processed. Skipping.")
                    
                    # Mark position as processed
                    processed_positions.add(position_key)
                    
                    file_handlers["is_correct"].write(f"{chrom}\t{pos_start}\t{pos_end}\t{metrics['is_correct'][i]}\n")
                    file_handlers["entropy"].write(f"{chrom}\t{pos_start}\t{pos_end}\t{metrics['entropy'][i]}\n")
                    file_handlers["highest_prob"].write(f"{chrom}\t{pos_start}\t{pos_end}\t{metrics['highest_prob'][i]}\n")
                    file_handlers["highest_prob_token"].write(f"{chrom}\t{pos_start}\t{pos_end}\t{metrics['highest_prob_token'][i]}\n")
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                continue
    
    # Close files
    for fh in file_handlers.values():
        fh.close()
    fasta.close()
    
    print(f"Results saved to {output_prefix}_*.bed/bedgraph")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process pre-computed model predictions from h5 file and generate metrics files."
    )
    parser.add_argument("--h5_path", type=str, required=True,
                       help="Path to h5 file with logits")
    parser.add_argument("--genome_path", type=str, required=True,
                       help="Path to hg38 genome FASTA file")
    parser.add_argument("--output_prefix", type=str, required=True,
                       help="Prefix for output files")
    parser.add_argument("--token_map", type=str, required=True, choices=["evo2", "caduceus"],
                        help="Predefined token map to use: 'evo2' or 'caduceus'")

    args = parser.parse_args()

    # Predefined token maps
    if args.token_map == "evo2":
        # Evo2: {"A":65, "C":67, "G":71, "T":84}
        token_map = {65: "A", 67: "C", 71: "G", 84: "T"}
    elif args.token_map == "caduceus":
        # Caduceus: A: tensor([7]), T: tensor([10]), C: tensor([8]), G: tensor([9])
        # We'll just use the integer values.
        token_map = {7: "A", 10: "T", 8: "C", 9: "G"}
    else:
        raise ValueError("Unknown token map specified. Choose 'evo2' or 'caduceus'.")
    
    return args, token_map


def main():
    args, token_map = parse_args()
    
    process_h5_file(
        h5_path=args.h5_path,
        genome_path=args.genome_path,
        token_map=token_map,
        output_prefix=args.output_prefix
    )


if __name__ == "__main__":
    main()


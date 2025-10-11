import os
import torch
from transformers import AutoTokenizer
from expression_dataset_tss_shifted import ExpressionDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def create_dataset(split='train', shift=10000):
    """Create ExpressionDataset instance for specified split"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")
    
    # Dataset parameters
    dataset_params = {
        'forward_intervals_path': f"{HOME_PATH}/GENA_LM/downstream_tasks/expression_prediction/intervals/human.{split}.forward.csv",
        'reverse_intervals_path': f"{HOME_PATH}/GENA_LM/downstream_tasks/expression_prediction/intervals/human.{split}.reverse.csv",
        'targets_path': f"{HOME_PATH}/GENA_LM/downstream_tasks/expression_prediction/datasets/data/file_mappings/Expression_dataset_v1_csv_file_mappings.csv",
        'gen_max_seq_len': 1008,
        'num_before': 250,
        'genome': f"{HOME_PATH}/GENA_LM/downstream_tasks/expression_prediction/datasets/data/genomes/hg38/hg38.fa",
        'gen_tokenizer': tokenizer,
        'text_data_path': f"{HOME_PATH}/GENA_LM/downstream_tasks/expression_prediction/datasets/data/file_mappings/full_combined_file_mappings_embeddings.pkl",
        'tpm': 'csv',
        'shift': shift,
        'transform_targets_tpm': lambda x: np.log(x + 0.01)  # log transform with pseudocount
    }
    
    return ExpressionDataset(**dataset_params)

def save_to_bed(dataset, output_dir, split):
    """Save dataset outputs to three file types per cell type:
    - *.TPM.bedgraph: TPM values
    - *.id.bed: Gene IDs with strand
    - *.seq.bed: Sequences with strand
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get cell type IDs from dataset
    cell_types = dataset.selected_keys_chunks[0]  # Get cell types from first chunk
    
    # Initialize dictionaries to store DataFrames for each cell type and file type
    cell_type_tpm_dfs = {cell_type: [] for cell_type in cell_types}
    cell_type_id_dfs = {cell_type: [] for cell_type in cell_types}
    cell_type_seq_dfs = {cell_type: [] for cell_type in cell_types}
    
    # Process each sample in the dataset with progress bar
    for i in tqdm(range(len(dataset)), desc=f"Processing {split} dataset"):
        sample = dataset[i]
        
        # Get sequence from input_ids
        sequence = dataset.gen_tokenizer.decode(sample['input_ids'].tolist())
        
        # Get TPM values and cell types for this chunk
        tpm_values = sample['tpm'].numpy()
        chunk_cell_types = sample['selected_keys']
        
        # Create rows for each non-NaN TPM value
        for j, (tpm, cell_type) in enumerate(zip(tpm_values, chunk_cell_types)):
            if not np.isnan(tpm):
                # TPM bedgraph format (4 columns: chrom, start, end, score)
                tpm_row = {
                    'chrom': sample['chrom'],
                    'start': sample['start'],
                    'end': sample['end'],
                    'score': tpm
                }
                cell_type_tpm_dfs[cell_type].append(tpm_row)
                
                # Gene ID bed format (6 columns: chrom, start, end, name, score, strand)
                id_row = {
                    'chrom': sample['chrom'],
                    'start': sample['start'],
                    'end': sample['end'],
                    'name': sample['gene_id'][j],
                    'score': 0,  # Using 0 as default score
                    'strand': sample['strand'] if 'strand' in sample else '.'  # Add strand if available
                }
                cell_type_id_dfs[cell_type].append(id_row)
                
                # Sequence bed format (6 columns: chrom, start, end, name, score, strand)
                seq_row = {
                    'chrom': sample['chrom'],
                    'start': sample['start'],
                    'end': sample['end'],
                    'name': sequence,
                    'score': 0,  # Using 0 as default score
                    'strand': sample['strand'] if 'strand' in sample else '.'  # Add strand if available
                }
                cell_type_seq_dfs[cell_type].append(seq_row)
    
    # Save separate files for each cell type and file type
    for cell_type in cell_types:
        # Save TPM bedgraph
        if cell_type_tpm_dfs[cell_type]:
            df_tpm = pd.DataFrame(cell_type_tpm_dfs[cell_type])
            output_file = os.path.join(output_dir, f"{split}_expression_{cell_type}.TPM.bedgraph")
            df_tpm.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"Saved {len(df_tpm)} TPM entries for cell type {cell_type} to {output_file}")
        
        # Save gene ID bed
        if cell_type_id_dfs[cell_type]:
            df_id = pd.DataFrame(cell_type_id_dfs[cell_type])
            output_file = os.path.join(output_dir, f"{split}_expression_{cell_type}.id.bed")
            df_id.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"Saved {len(df_id)} gene ID entries for cell type {cell_type} to {output_file}")
        
        # Save sequence bed
        if cell_type_seq_dfs[cell_type]:
            df_seq = pd.DataFrame(cell_type_seq_dfs[cell_type])
            output_file = os.path.join(output_dir, f"{split}_expression_{cell_type}.seq.bed")
            df_seq.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"Saved {len(df_seq)} sequence entries for cell type {cell_type} to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create bed files from expression dataset')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'valid'],
                      help='Dataset split to process (train or valid)')
    parser.add_argument('--shift', type=int, default=10000,
                      help='Shift value for TSS (default: 10000)')
    args = parser.parse_args()

    # Create output directory
    output_dir = f"{HOME_PATH}/GENA_LM/runs/bed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and process dataset
    print(f"Processing {args.split} dataset...")
    dataset = create_dataset(split=args.split, shift=args.shift)
    save_to_bed(dataset, output_dir, args.split)

if __name__ == "__main__":
    main() 
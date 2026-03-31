#!/usr/bin/env python3
import argparse
import pandas as pd
import yaml
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CRE prediction model config from a table.")
    
    # Input/Output arguments
    parser.add_argument("--table_path", type=str, default='metadata/config_entries.csv', help="Path to the input table (CSV or TSV).")
    parser.add_argument("--output_path", type=str, default='configs/config_base.yaml', help="Path to save the output YAML config.")
    
    # Dataset parameters with defaults based on the provided example
    # Note: String defaults that are None in the example are set to None here.
    parser.add_argument("--inputType", type=str, default='TSS', help="Input type (e.g., TSS).")
    parser.add_argument("--unknown_taxon_prob", type=float, default=0.15, help="Probability for unknown taxon.")
    parser.add_argument("--min_cre_number", type=int, default=None, help="Minimum CRE number.")
    parser.add_argument("--min_cre_coverage", type=int, default=None, help="Minimum CRE coverage.")
    parser.add_argument("--cre_bed", type=str, default=None, help="Path to CRE bed file.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--tokenizer", type=str, default='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/data/tokenizers/t2t_1000h_multi_32k', help="Path to tokenizer.")
    parser.add_argument("--token_len_for_fetch", type=int, default=10, help="Token length for fetch.")
    parser.add_argument("--cache_dir", type=str, default='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/cache', help="Cache directory.")
    
    return parser.parse_args()

def load_table(path):
    """Loads a CSV or TSV file into a pandas DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.csv']:
        return pd.read_csv(path)
    elif ext in ['.tsv', '.txt']:
        return pd.read_csv(path, sep='\t')
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    else:
        # Default to CSV if extension is unknown or missing
        print(f"Warning: Unknown extension '{ext}', attempting to read as CSV.")
        return pd.read_csv(path)

def clean_value(val):
    """Converts pandas NaN/NaT to Python None for YAML compatibility."""
    if pd.isna(val):
        return None
    return val

def main():
    args = parse_args()
    
    # Load the table
    try:
        df = load_table(args.table_path)
    except Exception as e:
        print(f"Error loading table: {e}")
        sys.exit(1)

    # Required columns check
    required_cols = ['annotation', 'genomePath', 'chrom_file', 'taxon', 'split']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in table: {missing_cols}")
        sys.exit(1)

    configs = []

    # Iterate over each row to create a dataset entry
    for idx, row in df.iterrows():
        # Determine the key name based on split
        split_val = clean_value(row.get('split'))
        if split_val:
            dataset_key = f"{split_val}_dataset_{row.get('dataset_name')}"
            #dataset_key = f"{split_val}_dataset"
        else:
            # Default to train_dataset if split is not specified
            dataset_key = "train_dataset"
            
        # Handle potential duplicate keys if multiple rows lack split or have same split
        #if dataset_key in config_dict:
        #    print(f"Warning: Duplicate dataset key '{dataset_key}' found (row {idx}). Appending index.")
        #    dataset_key = f"{dataset_key}_{idx}"

        # Build the dataset configuration
        # Start with the fixed target
        dataset_config = {
            "_target_": "CRE_dataset.CreDataset"
        }

        # Map table columns to config keys
        # Note: genomePath in table maps to genome in config
        dataset_config["genome"] = clean_value(row.get('genomePath'))
        dataset_config["chrom_file"] = clean_value(row.get('chrom_file'))
        dataset_config["taxon"] = clean_value(row.get('taxon'))
        dataset_config["annotation"] = clean_value(row.get('annotation'))
        
        # Add parameters from argparse (defaults or user overrides)
        dataset_config["inputType"] = args.inputType
        dataset_config["unknown_taxon_prob"] = args.unknown_taxon_prob
        dataset_config["min_cre_number"] = args.min_cre_number
        dataset_config["min_cre_coverage"] = args.min_cre_coverage
        dataset_config["cre_bed"] = args.cre_bed
        dataset_config["max_seq_len"] = args.max_seq_len
        dataset_config["tokenizer"] = args.tokenizer
        dataset_config["token_len_for_fetch"] = args.token_len_for_fetch
        dataset_config["cache_dir"] = args.cache_dir

        # Add to main config
        configs.append({dataset_key:dataset_config})

    # Write to YAML
    try:
        with open(args.output_path, 'w') as f:
            for config in configs:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"Configuration successfully written to {args.output_path}")
    except Exception as e:
        print(f"Error writing YAML file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
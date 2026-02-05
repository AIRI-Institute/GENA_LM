import os
import yaml
import argparse
from pathlib import Path

def generate_yaml_config(split_dir_path, output_path=""):
    """
    Generate YAML configuration file from the split directory structure.
    
    Args:
        split_dir_path: Path to the split_name directory
        home_path: HOME_PATH variable value (default: /home)
        output_file: Output YAML file name
    """
    split_dir = Path(split_dir_path)
    
    # Check if the directory exists
    if not split_dir.exists():
        print(f"Error: Directory {split_dir_path} does not exist")
        return
    
    # Get split name from the directory name
    split_name = split_dir.name
    
    # Path to intervals directory
    intervals_dir = split_dir / "intervals"
    
    if not intervals_dir.exists():
        print(f"Error: Intervals directory not found at {intervals_dir}")
        return
    
    config = {}
    
    # Iterate through each ncbi_taxonomy_id directory
    for tax_id_dir in intervals_dir.iterdir():
        if not tax_id_dir.is_dir():
            continue
        
        ncbi_taxonomy_id = tax_id_dir.name
        
        # Create configurations for train, valid, and test datasets
        for dataset_type in ["train", "valid"]:
            forward_file = tax_id_dir / f"{ncbi_taxonomy_id}.{dataset_type}.forward.tsv"
            reverse_file = tax_id_dir / f"{ncbi_taxonomy_id}.{dataset_type}.reverse.tsv"
            
            if forward_file.exists() and reverse_file.exists():
                # Create the config entry name
                entry_name = f"{dataset_type}_dataset_{split_name}_{ncbi_taxonomy_id}"
                
                # Get absolute paths for the interval files
                forward_intervals_path = forward_file.absolute()
                reverse_intervals_path = reverse_file.absolute()
                
                # Create the config entry
                config[entry_name] = {
                    "_target_": "downstream_tasks.expression_prediction.expression_dataset_fr_embed.ExpressionDataset",
                    "genome": f"${{HOME_PATH}}/GENA_LM/downstream_tasks/expression_prediction/datasets/data/genomes/{ncbi_taxonomy_id}/{ncbi_taxonomy_id}.fa",
                    "targets_path": f"/scratch/tsoies-Expression/GENA_LM/downstream_tasks/expression_prediction/datasets/data/{ncbi_taxonomy_id}_file_mappings.csv",
                    "desc_embeddings_path": f"${{HOME_PATH}}/GENA_LM/downstream_tasks/expression_prediction/datasets/data/{ncbi_taxonomy_id}_embeddings.pkl",
                    "forward_intervals_path": str(forward_intervals_path).replace('valid', 'test'),
                    "reverse_intervals_path": str(reverse_intervals_path).replace('valid', 'test'),
                    "tpm": "tpm",
                    "desc_embedding_dim": 768,
                    "use_cached_embeddings": False
                }
            elif forward_file.exists() or reverse_file.exists():
                # Only one of the files exists
                missing_type = "reverse" if forward_file.exists() else "forward"
                print(f"Warning: Missing {missing_type} {dataset_type} file for {ncbi_taxonomy_id}")
    

    output_file = Path(output_path) / (split_name+'.yaml')
    # Write to YAML file
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration written to {output_file}")
    print(f"Found {len(config)} dataset configurations")
    
    # Print summary
    dataset_types = {}
    for key in config.keys():
        dataset_type = key.split('_')[0]  # train, valid, or test
        dataset_types[dataset_type] = dataset_types.get(dataset_type, 0) + 1
    
    print("\nSummary:")
    for dtype, count in dataset_types.items():
        print(f"  {dtype}: {count} configurations")
    
    # Print example paths for verification
    if config:
        first_key = list(config.keys())[0]
        print(f"\nExample configuration for {first_key}:")
        print(f"  forward_intervals_path: {config[first_key]['forward_intervals_path']}")
        print(f"  reverse_intervals_path: {config[first_key]['reverse_intervals_path']}")

def main():
    parser = argparse.ArgumentParser(description="Generate YAML config from intervals directory structure")
    parser.add_argument("split_dir", help="Path to the split_name directory")
    parser.add_argument("--outdir", default="", help="Output dir")
    
    args = parser.parse_args()
    
    generate_yaml_config(args.split_dir, args.outdir)

if __name__ == "__main__":
    main()
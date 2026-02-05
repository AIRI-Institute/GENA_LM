import polars as pl
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Optional

def split_csv_files(input_dir: str, output_dir: Optional[str] = None, 
                   verbose: bool = True) -> Dict[str, int]:
    """
    Split CSV files into multiple files, each with one line of data.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Output directory (default: same as input_dir/split)
        verbose: Print progress information
    
    Returns:
        Dictionary with split name -> count of files created
    """
    input_path = Path(input_dir)
    
    # Set output directory
    if output_dir is None:
        output_dir = input_path / "split"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Input directory: {input_path}")
        print(f"Output directory: {output_path}")
    
    # Find CSV files with pattern
    csv_files = list(input_path.glob("*_expr_file_mappings.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir} matching pattern '*_expr_file_mappings.csv'")
        return {}
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files to process")
    
    results = {}
    
    for csv_file in csv_files:
        if verbose:
            print(f"\nProcessing: {csv_file.name}")
        
        # Extract split name from filename
        split_name = csv_file.name.replace("_expr_file_mappings.csv", "")
        
        try:
            # Read CSV file using polars (faster) or pandas as fallback
            try:
                df = pl.read_csv(csv_file)
            except Exception as e:
                if verbose:
                    print(f"  Polars failed, trying pandas: {e}")
                df_pd = pd.read_csv(csv_file)
                df = pl.from_pandas(df_pd)
            
            total_rows = len(df)
            
            if verbose:
                print(f"  Found {total_rows} rows in file")
                print(f"  Columns: {df.columns}")
            
            # Check for required columns
            required_cols = ['id', 'genome', 'dataset_description']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  Warning: Missing columns {missing_cols} in {csv_file.name}")
                continue
            
            # Group by genome and create separate files
            files_created = 0
            
            for genome_value in df['genome'].unique().to_list():
                # Filter rows for this genome
                genome_df = df.filter(pl.col('genome') == genome_value)
                
                genome = genome_df['genome'].item()
                
                output_filename = f"{genome}_file_mappings.csv"
                output_file = output_path / output_filename
                
                # Write to CSV
                genome_df.write_csv(output_file)
                files_created += 1
                
                if verbose:
                    print(f"    Created: {output_filename} ({len(genome_df)} rows)")
            
            results[split_name] = files_created
            
            if verbose:
                print(f"  Created {files_created} files for {split_name}")
        
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Split CSV files into multiple files based on genome"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: input_dir/split)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Check input directory exists
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    # Process files
    results = split_csv_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        verbose=args.verbose and not args.quiet
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        total_files = sum(results.values())
        total_splits = len(results)
        
        print(f"Total splits processed: {total_splits}")
        print(f"Total files created: {total_files}")
        print("\nDetails by split:")
        
        for split_name, file_count in sorted(results.items()):
            print(f"  {split_name}: {file_count} files")
    else:
        print("No files were processed.")
    
    return 0

if __name__ == "__main__":
    exit(main())
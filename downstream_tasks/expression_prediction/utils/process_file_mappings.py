import polars as pl
import argparse
from pathlib import Path

# Mapping from ncbi genome id to scientific name
NAMING_DICT = {
    'GCF_000001405.40': 'Homo_sapiens',
    'GCF_000001635.27': 'Mus_musculus',
    'GCF_036323735.1': 'Rattus_norvegicus',
    'GCF_000003025.6': 'Sus_scrofa',
    'GCF_002263795.3': 'Bos_taurus',
    'GCF_011100685.1': 'Canis_lupus',
    'GCF_049350105.2': 'Macaca_mulatta',
    'GCF_016772045.2': 'Ovis_aries',
    'GCF_964237555.1': 'Oryctolagus_cuniculus',
    'GCF_041296265.1': 'Equus_caballus'
}

def update_dataset_description(input_csv, output_csv=None, verbose=True):
    """
    Update dataset_description column with species names based on genome ID.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (if None, overwrite input)
        verbose: Print summary information
    """
    # Read the CSV file
    if verbose:
        print(f"Reading input file: {input_csv}")
    
    try:
        df = pl.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False
    
    # Check if required columns exist
    required_columns = ['id', 'genome', 'dataset_description']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns}")
        return False
    
    if verbose:
        print(f"Loaded {len(df)} rows with columns: {df.columns}")
    
    
    # Update dataset_description column
    df_updated = df.with_columns(
        dataset_description=pl.col('genome').replace_strict(NAMING_DICT)
    )
    
    # Count how many rows were updated
    original_genomes = df["genome"].unique().to_list()
    updated_count = df_updated.filter(
        pl.col("genome").is_in(list(NAMING_DICT.keys())) &
        (pl.col("dataset_description") != df["dataset_description"])
    ).height
    
    # Determine output path
    if output_csv is None:
        output_csv = input_csv
        if verbose:
            print(f"Overwriting input file: {input_csv}")
    
    # Write to output CSV
    try:
        df_updated.write_csv(output_csv)
        if verbose:
            print(f"Successfully wrote {len(df_updated)} rows to: {output_csv}")
            print(f"Updated dataset_description for {updated_count} rows")
            
            # Print summary of genomes found
            print("\nGenome distribution in dataset:")
            genome_counts = df_updated.group_by("genome").agg(
                pl.count().alias("count"),
                pl.col("dataset_description").first().alias("species")
            ).sort("count", descending=True)
            print(genome_counts)
            
            # Show mapping that was applied
            print("\nMapping applied:")
            for genome_id, species_name in NAMING_DICT.items():
                count = df_updated.filter(pl.col("genome") == genome_id).height
                if count > 0:
                    print(f"  {genome_id} -> {species_name}: {count} rows")
            
            # Show genomes not in mapping
            unmapped_genomes = df_updated.filter(
                ~pl.col("genome").is_in(list(NAMING_DICT.keys()))
            ).select("genome").unique()
            
            if unmapped_genomes.height > 0:
                print(f"\nWarning: {unmapped_genomes.height} genomes not in naming dictionary:")
                for row in unmapped_genomes.iter_rows():
                    print(f"  {row[0]}")
    
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Update dataset_description column with species names based on genome ID"
    )
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output CSV file (if not specified, overwrites input)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_csv}")
        return
    
    success = update_dataset_description(
        input_csv=args.input_csv,
        output_csv=args.output,
        verbose=not args.quiet
    )
    
    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed!")

if __name__ == "__main__":
    main()
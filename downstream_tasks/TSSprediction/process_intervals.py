#!/usr/bin/env python3
"""
Extract top N CRE intervals from CSV files and produce forward/reverse strand files.

Usage:
    python script.py input_dir output_dir N
"""

import polars as pl
import argparse
from pathlib import Path

def process_file(file_path: Path, output_dir: Path, n: int):
    # Load CSV
    df = pl.read_csv(file_path)

    # Step 1: sort by cre_prob descending, keep top N
    df_top = df.sort("cre_prob", descending=True).head(n)

    # Step 2: sort by chrom and start
    df_top = df_top.sort(["chrom", "start"])

    # Step 3: add gene_id column "cre_1" ... "cre_N"
    df_top = df_top.with_row_count("idx")
    df_top = df_top.with_columns(
        (pl.lit("cre_") + (pl.col("idx") + 1).cast(pl.Utf8)).alias("gene_id")
    ).drop("idx")

    # Prepare output filenames from input stem (e.g., "human_enhancers.csv")
    stem = file_path.stem
    if "_" in stem:
        species, name = stem.split("_", 1)
    else:
        species, name = stem, ""

    # Forward strand
    fwd = df_top.select(
        (pl.col("gene_id") + pl.lit("_fwd")).alias("gene_id"),
        pl.col("chrom").alias("chromosome"),
        pl.col("start").alias("TSS"),
        pl.col("end").alias("TES"),
        "cre_prob",
    )
    fwd_path = output_dir / f"{species}_{name}_forward.csv"
    fwd.write_csv(fwd_path)

    # Reverse strand
    rev = df_top.select(
        (pl.col("gene_id") + pl.lit("_rev")).alias("gene_id"),
        pl.col("chrom").alias("chromosome"),
        pl.col("start").alias("TES"),
        pl.col("end").alias("TSS"),
        "cre_prob",
    )
    rev_path = output_dir / f"{species}_{name}_reverse.csv"
    rev.write_csv(rev_path)

def main():
    parser = argparse.ArgumentParser(
        description="Extract top N CRE intervals and produce forward/reverse files."
    )
    parser.add_argument("input_dir", help="Directory containing input CSV files")
    parser.add_argument("output_dir", help="Directory where output files will be written")
    parser.add_argument("N", type=int, help="Number of top intervals to select")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process every .csv file in the input directory
    for csv_file in input_path.glob("*.csv"):
        print(f"Processing {csv_file.name}...")
        process_file(csv_file, output_path, args.N)

    print("Done.")

if __name__ == "__main__":
    main()
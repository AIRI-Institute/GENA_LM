import pandas as pd
import numpy as np
import os

# Read the BED file
cell_name = 'ENCFF035CWS'
genes_metadata = pd.read_csv("../datasets/data/genomes/hg38/hg38_genes.csv")

# Read the ground truth and predicted expression files
prefix = "/mnt/nfs_dna/aspeedok/github/results/protein_coding_train_shift_10000"
ground_truth = pd.read_csv(f'{prefix}_Expression_dataset_v1_GRCh38_csv dataset_true.csv')
predicted = pd.read_csv(f'{prefix}_Expression_dataset_v1_GRCh38_csv dataset_pred.csv')

# Function to create bedgraph file
def create_bedgraph(genes_metadata, expression_df, output_file):
    # Merge BED coordinates with expression values
    merged_df = pd.merge(genes_metadata, expression_df, on='gene_id', how='inner', validate='one_to_one')
    
    # Select required columns and sort
    merged_df["bg_start"] = merged_df["TSS"]
    merged_df["bg_end"] = merged_df["TSS"]
    merged_df.loc[merged_df["gene_strand"]=="+","bg_end"] = merged_df["TSS"] + 10000
    merged_df.loc[merged_df["gene_strand"]=="-","bg_start"] = merged_df["TSS"] - 10000
    bedgraph_df = merged_df[['chromosome', 'bg_start', 'bg_end', f'{cell_name}']].sort_values(['chromosome', 'bg_start'])
    
    # Write to bedgraph file
    # Write header first
    with open(output_file, 'w') as f:
        f.write('track type=bedGraph visibility=full priority=20 maxHeightPixels=80:80:80\n')
    
    # Append data
    bedgraph_df.to_csv(output_file, sep='\t', header=False, index=False, mode='a')

# Create bedgraph files
prefix = os.path.basename(prefix)
create_bedgraph(genes_metadata, ground_truth, f'{prefix}_ground_truth_{cell_name}.bedgraph')
create_bedgraph(genes_metadata, predicted, f'{prefix}_predicted_{cell_name}.bedgraph') 
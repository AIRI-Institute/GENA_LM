import pandas as pd
import numpy as np

# Read the BED file
cell_name = 'ENCFF035CWS'
bed_df = pd.read_csv(f'valid_expression_{cell_name}.id.bed', sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'gene_id', 'score', 'strand'])

# Read the ground truth and predicted expression files
ground_truth = pd.read_csv('Expression_dataset_v1_GRCh38_csv dataset_true.csv')
predicted = pd.read_csv('shift_10000_Expression_dataset_v1_GRCh38_csv dataset_pred.csv')

# Function to create bedgraph file
def create_bedgraph(bed_df, expression_df, output_file):
    # Merge BED coordinates with expression values
    merged_df = pd.merge(bed_df, expression_df, on='gene_id', how='inner', validate='one_to_one')
    
    # Select required columns and sort
    bedgraph_df = merged_df[['chrom', 'start', 'end', f'{cell_name}']].sort_values(['chrom', 'start'])
    
    # Write to bedgraph file
    bedgraph_df.to_csv(output_file, sep='\t', header=False, index=False)

# Create bedgraph files
create_bedgraph(bed_df, ground_truth, f'ground_truth_{cell_name}.bedgraph')
create_bedgraph(bed_df, predicted, f'predicted_{cell_name}.bedgraph') 
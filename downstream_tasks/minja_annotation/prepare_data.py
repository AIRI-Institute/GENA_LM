import os
import gffutils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--genome_dir", type=str, default="../../data/annotation")
parser.add_argument("--base_name", type=str, default="gencode.v48.annotation.gtf")
parser.add_argument("--fasta_name", type=str, default="hg38.fa")
args = parser.parse_args()

# Define paths
gff_path = os.path.join(args.genome_dir, args.base_name)
db_path = os.path.join(args.genome_dir, f"{args.base_name}.db")

print (f"Creating GFF database from {gff_path} to {db_path}")

try:
	# Create or load the database
	if not os.path.exists(db_path):
		db = gffutils.create_db(
			gff_path,
			dbfn=db_path,
			disable_infer_transcripts=True,
			disable_infer_genes=True,
			merge_strategy="create_unique"
		)
	else:
		db = gffutils.FeatureDB(db_path)
except Exception as e:
	print ("Error creating or loading the gff database")
	raise e

print ("GFF database created or loaded")

import pysam
fasta_path = os.path.join(args.genome_dir, args.fasta_name)
try:
	reference_lengths = pysam.FastaFile(fasta_path).references
except Exception as e:
	print ("Error fetching fasta")
	raise e

print ("FASTA file fetched")
print ("Data setup complete and verified")
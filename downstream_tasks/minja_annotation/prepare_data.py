import os
import gffutils

# Define paths
genome_dir = "../../data/annotation"
base_name = "gencode.v48.annotation.gtf"
gff_path = os.path.join(genome_dir, base_name)
db_path = os.path.join(genome_dir, f"{base_name}.db")

try:
	# Create or load the database
	if not os.path.exists(db_path):
		db = gffutils.create_db(
			gff_path,
			dbfn=db_path,
			disable_infer_transcripts=True,
			disable_infer_genes=True
		)
	else:
		db = gffutils.FeatureDB(db_path)
except Exception as e:
	print ("Error creating or loading the gff database")
	raise e

print ("GFF database created or loaded")

import pysam
fasta_path = genome_dir + "/hg38.fa"
try:
	fasta = pysam.FastaFile(fasta_path)
	fasta.fetch("chr1", 0, 1000000)
except Exception as e:
	print ("Error fetching fasta")
	raise e

print ("FASTA file fetched")
print ("Data setup complete and verified")
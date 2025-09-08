#!/bin/bash

# Download the unzip Gencode annotation file
mkdir -p ../../data/annotation
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gtf.gz -O ../../data/annotation/gencode.v48.annotation.gtf.gz
gunzip ../../data/annotation/gencode.v48.annotation.gtf.gz

# Download the hg38.fa file
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O ../../data/annotation/hg38.fa.gz
gunzip ../../data/annotation/hg38.fa.gz

# For T2T
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/914/755/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz -O ../../data/annotation/T2T-CHM13v2.0_genomic.fna.gz
gunzip ../../data/annotation/T2T-CHM13v2.0_genomic.fna.gz

# TODO: make gff download more generalizable
ln -s /mnt/nfs_protein/shadskiy/GENA/biodata/genomic.gff ../../data/annotation/GCF_009914755.1_T2T-CHM13v2.0.gff

# Run the prepare_data.py script
python3 minja_annotation/prepare_data.py --genome_dir ../../data/annotation --base_name gencode.v48.annotation.gtf --fasta_name hg38.fa
python3 prepare_data.py --genome_dir ../../data/annotation --base_name GCF_009914755.1_T2T-CHM13v2.0.gff --fasta_name T2T-CHM13v2.0_genomic.fna
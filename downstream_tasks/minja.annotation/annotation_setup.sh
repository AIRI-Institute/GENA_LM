#!/bin/bash

# Download the unzip Gencode annotation file
mkdir -p ../../data/annotation
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gtf.gz -O ../../data/annotation/gencode.v48.annotation.gtf.gz
gunzip ../../data/annotation/gencode.v48.annotation.gtf.gz

# Download the hg38.fa file
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O ../../data/annotation/hg38.fa.gz
gunzip ../../data/annotation/hg38.fa.gz

# Run the prepare_data.py script
python3 minja.annotation/prepare_data.py
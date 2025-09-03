#!/bin/bash

# Download the unzip Gencode annotation file
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gtf.gz -O ../../data/gencode.v48.annotation.gtf.gz
gunzip ../../data/gencode.v48.annotation.gtf.gz

# Download the hg38.fa file
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O ../../data/hg38.fa.gz
gunzip ../../data/hg38.fa.gz

# Run the prepare_data.py script
python3 downstream_tasks/minja.annotation/prepare_data.py

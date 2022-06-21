# download RefSeq .FASTA
curl https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.fna.gz --output genomes/human_GRCh38.p13.fna.gz
# unpack
# gzip -dk genomes/human_GRCh38.p13.fna.gz
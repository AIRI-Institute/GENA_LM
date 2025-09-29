#!/bin/bash

# Function to check if file exists and download if not
download_if_not_exists() {
    local url="$1"
    local output_path="$2"
    local description="$3"
    local unzipped_path="$4"  # Optional: path to unzipped version
    
    # Check if either the compressed or unzipped file exists
    if [ -f "$output_path" ]; then
        echo "File already exists: $output_path ($description)"
    elif [ -n "$unzipped_path" ] && [ -f "$unzipped_path" ]; then
        echo "Unzipped file already exists: $unzipped_path ($description) - skipping download"
    else
        echo "Downloading $description to $output_path..."
        wget "$url" -O "$output_path"
    fi
}

# Function to check if file exists and unzip if not
unzip_if_not_exists() {
    local zip_path="$1"
    local unzipped_path="$2"
    local description="$3"
    
    if [ -f "$unzipped_path" ]; then
        echo "Unzipped file already exists: $unzipped_path ($description)"
    else
        echo "Unzipping $description..."
        gunzip "$zip_path"
    fi
}

# Create annotation directory
mkdir -p ../../data/annotation

# Download the Gencode annotation file
download_if_not_exists "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gtf.gz" "../../data/annotation/gencode.v48.annotation.gtf.gz" "Gencode annotation file" "../../data/annotation/gencode.v48.annotation.gtf"
unzip_if_not_exists "../../data/annotation/gencode.v48.annotation.gtf.gz" "../../data/annotation/gencode.v48.annotation.gtf" "Gencode annotation file"

# Download the hg38.fa file
download_if_not_exists "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" "../../data/annotation/hg38.fa.gz" "hg38 reference genome" "../../data/annotation/hg38.fa"
unzip_if_not_exists "../../data/annotation/hg38.fa.gz" "../../data/annotation/hg38.fa" "hg38 reference genome"

# For T2T
download_if_not_exists "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/914/755/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz" "../../data/annotation/T2T-CHM13v2.0_genomic.fna.gz" "T2T-CHM13v2.0 reference genome" "../../data/annotation/T2T-CHM13v2.0_genomic.fna"
unzip_if_not_exists "../../data/annotation/T2T-CHM13v2.0_genomic.fna.gz" "../../data/annotation/T2T-CHM13v2.0_genomic.fna" "T2T-CHM13v2.0 reference genome"

# TODO: make gff download more generalizable
if [ -f "../../data/annotation/GCF_009914755.1_T2T-CHM13v2.0.gff" ]; then
    echo "GFF file already exists: ../../data/annotation/GCF_009914755.1_T2T-CHM13v2.0.gff"
else
    echo "Creating symbolic link for GFF file..."
    ln -s /mnt/nfs_protein/shadskiy/GENA/biodata/genomic.gff ../../data/annotation/GCF_009914755.1_T2T-CHM13v2.0.gff
fi

# Download variants
mkdir -p ../../data/annotation/1kg_variants
outdir="../../data/annotation/1kg_variants"

for chr in {1..22}; do
    vcf_file="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    tbi_file="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz.tbi"
    
    download_if_not_exists "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/${vcf_file}" "${outdir}/${vcf_file}" "1000 Genomes VCF for chromosome ${chr}"
    download_if_not_exists "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/${tbi_file}" "${outdir}/${tbi_file}" "1000 Genomes TBI index for chromosome ${chr}"
done

vcf_file="1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz"
output_vcf_file=${outdir}/"1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"

tbi_file="1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz.tbi"
output_tbi_file=${outdir}/"1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.vcf.gz.tbi"

download_if_not_exists "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/${vcf_file}" "$output_vcf_file" "1000 Genomes VCF for chromosome ${chr}"
download_if_not_exists "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/${tbi_file}" "$output_tbi_file" "1000 Genomes TBI index for chromosome ${chr}"

# Run the prepare_data.py script
python3 minja_annotation/prepare_data.py --genome_dir ../../data/annotation --base_name gencode.v48.annotation.gtf --fasta_name hg38.fa
python3 prepare_data.py --genome_dir ../../data/annotation --base_name GCF_009914755.1_T2T-CHM13v2.0.gff --fasta_name T2T-CHM13v2.0_genomic.fna
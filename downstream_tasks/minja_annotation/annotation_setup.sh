#!/bin/bash

###############################################################################
# Helper functions
###############################################################################

# Function to check if file exists and download if not (UNCHANGED)
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

# Function to check if file exists and unzip if not (UNCHANGED)
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

# Make sure NCBI Datasets CLI is available (NEW)
require_datasets_cli() {
    if ! command -v datasets >/dev/null 2>&1; then
        echo "ERROR: 'datasets' command not found in your PATH." >&2
        echo "Please install NCBI Datasets CLI:" >&2
        echo "  https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/" >&2
        exit 1
    fi
}

# Download one non-human genome using NCBI Datasets into its own folder (NEW)
download_genome_with_datasets() {
    local accession="$1"         # e.g. GCF_000952055.2
    local base_dir="$2"          # e.g. ../../data/annotation/38_more_genomes

    local acc_dir="${base_dir}/${accession}"
    mkdir -p "$acc_dir"

    local fasta_dest="${acc_dir}/genome.fna"
    # We'll decide GFF dest name by extension
    local gff_dest_gff3="${acc_dir}/annotation.gff3"
    local gff_dest_gff="${acc_dir}/annotation.gff"

    # Skip if we already have both genome and some annotation file
    if [ -f "$fasta_dest" ] && { [ -f "$gff_dest_gff3" ] || [ -f "$gff_dest_gff" ]; }; then
        echo "Genome and annotation already exist for ${accession} in ${acc_dir}"
        return 0
    fi

    local zip_path="${acc_dir}/${accession}_dataset.zip"

    echo "Downloading ${accession} genome + GFF3 via NCBI Datasets..."
    datasets download genome accession "$accession" \
        --annotated \
        --include genome,gff3 \
        --filename "$zip_path"

    echo "Unzipping dataset for ${accession} into ${acc_dir}..."
    unzip -o "$zip_path" -d "$acc_dir" >/dev/null

    # Locate genomic FASTA and GFF/GFF3 inside this accession directory
    local fasta_src=""
    local gff_src=""

    fasta_src=$(find "$acc_dir" -type f -name "*${accession}*genomic.fna" | head -n 1)
    if [ -z "$fasta_src" ]; then
        fasta_src=$(find "$acc_dir" -type f -name "*_genomic.fna" | head -n 1)
    fi
    if [ -z "$fasta_src" ]; then
        fasta_src=$(find "$acc_dir" -type f -name "*.fna" | head -n 1)
    fi

    gff_src=$(find "$acc_dir" -type f \( -name "*.gff3" -o -name "*.gff" \) | head -n 1)

    if [ -z "$fasta_src" ] || [ -z "$gff_src" ]; then
        echo "ERROR: Could not locate FASTA or GFF file for ${accession} in ${acc_dir}." >&2
        echo "  FASTA: ${fasta_src:-not found}" >&2
        echo "  GFF:   ${gff_src:-not found}" >&2
        return 1
    fi

    echo "Standardizing files for ${accession} in ${acc_dir}..."
    cp "$fasta_src" "$fasta_dest"

    if [[ "$gff_src" == *.gff3 ]]; then
        cp "$gff_src" "$gff_dest_gff3"
        echo "  -> ${fasta_dest}"
        echo "  -> ${gff_dest_gff3}"
    else
        cp "$gff_src" "$gff_dest_gff"
        echo "  -> ${fasta_dest}"
        echo "  -> ${gff_dest_gff}"
    fi
}

###############################################################################
# Directories
###############################################################################

ANNOTATION_DIR="../../data/annotation"                  # where hg38 etc. live (UNCHANGED location)
MORE_GENOMES_DIR="${ANNOTATION_DIR}/38_more_genomes"    # NEW: all extra genomes live here

mkdir -p "$ANNOTATION_DIR" "$MORE_GENOMES_DIR"

###############################################################################
# Human (hg38 / Gencode) and T2T data – original behavior
###############################################################################

# Download the Gencode annotation file (UNCHANGED, just using $ANNOTATION_DIR)
download_if_not_exists \
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gtf.gz" \
    "${ANNOTATION_DIR}/gencode.v48.annotation.gtf.gz" \
    "Gencode annotation file" \
    "${ANNOTATION_DIR}/gencode.v48.annotation.gtf"

unzip_if_not_exists \
    "${ANNOTATION_DIR}/gencode.v48.annotation.gtf.gz" \
    "${ANNOTATION_DIR}/gencode.v48.annotation.gtf" \
    "Gencode annotation file"

# Download the hg38.fa file (UNCHANGED, just using $ANNOTATION_DIR)
download_if_not_exists \
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" \
    "${ANNOTATION_DIR}/hg38.fa.gz" \
    "hg38 reference genome" \
    "${ANNOTATION_DIR}/hg38.fa"

unzip_if_not_exists \
    "${ANNOTATION_DIR}/hg38.fa.gz" \
    "${ANNOTATION_DIR}/hg38.fa" \
    "hg38 reference genome"

# For T2T (UNCHANGED, just using $ANNOTATION_DIR)
download_if_not_exists \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/914/755/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz" \
    "${ANNOTATION_DIR}/T2T-CHM13v2.0_genomic.fna.gz" \
    "T2T-CHM13v2.0 reference genome" \
    "${ANNOTATION_DIR}/T2T-CHM13v2.0_genomic.fna"

unzip_if_not_exists \
    "${ANNOTATION_DIR}/T2T-CHM13v2.0_genomic.fna.gz" \
    "${ANNOTATION_DIR}/T2T-CHM13v2.0_genomic.fna" \
    "T2T-CHM13v2.0 reference genome"

# T2T GFF symlink (UNCHANGED, only using $ANNOTATION_DIR variable)
# TODO: make gff download more generalizable
if [ -f "${ANNOTATION_DIR}/GCF_009914755.1_T2T-CHM13v2.0.gff" ]; then
    echo "GFF file already exists: ${ANNOTATION_DIR}/GCF_009914755.1_T2T-CHM13v2.0.gff"
else
    echo "Creating symbolic link for GFF file..."
    ln -s /mnt/nfs_protein/shadskiy/GENA/biodata/genomic.gff \
          "${ANNOTATION_DIR}/GCF_009914755.1_T2T-CHM13v2.0.gff"
fi

###############################################################################
# 1000 Genomes variants – original behavior
###############################################################################

VARIANTS_DIR="${ANNOTATION_DIR}/1kg_variants"
mkdir -p "$VARIANTS_DIR"
outdir="$VARIANTS_DIR"

# Autosomes 1..22 (UNCHANGED)
for chr in {1..22}; do
    vcf_file="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    tbi_file="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz.tbi"

    download_if_not_exists \
        "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/${vcf_file}" \
        "${outdir}/${vcf_file}" \
        "1000 Genomes VCF for chromosome ${chr}"

    download_if_not_exists \
        "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/${tbi_file}" \
        "${outdir}/${tbi_file}" \
        "1000 Genomes TBI index for chromosome ${chr}"
done

# Chromosome X (same URLs as before, wording fixed)
vcf_file="1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz"
output_vcf_file="${outdir}/1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"

tbi_file="1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz.tbi"
output_tbi_file="${outdir}/1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.vcf.gz.tbi"

download_if_not_exists \
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/${vcf_file}" \
    "$output_vcf_file" \
    "1000 Genomes VCF for chromosome X"

download_if_not_exists \
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/${tbi_file}" \
    "$output_tbi_file" \
    "1000 Genomes TBI index for chromosome X"

###############################################################################
# New: 38 additional genomes in 38_more_genomes/<accession> folders
###############################################################################

ADDITIONAL_GENOMES=(
    GCF_000952055.2
    GCF_002263795.3
    GCF_000767855.1
    GCF_000002285.3
    GCF_000151735.1
    GCF_001604975.1
    GCF_000283155.1
    GCF_000276665.1
    GCF_000260355.1
    GCF_002940915.1
    GCF_000151885.1
    GCF_002288905.1
    GCF_000308155.1
    GCF_000002305.2
    GCF_018350175.1
    GCF_000247695.1
    GCF_000236235.1
    GCF_000280705.1
    GCF_000001905.1
    GCF_001458135.1
    GCF_000165445.2
    GCF_000317375.1
    GCF_000001635.26
    GCF_900095145.1
    GCF_002201575.1
    GCF_000292845.1
    GCF_000260255.1
    GCF_000321225.1
    GCF_009806435.1
    GCF_000181295.1
    GCF_016772045.2
    GCF_000956105.1
    GCF_003327715.1
    GCF_036323735.1
    GCF_000235385.1
    GCF_000181275.1
    GCF_000003025.6
    GCF_000243295.1
)

if [ "${#ADDITIONAL_GENOMES[@]}" -gt 0 ]; then
    require_datasets_cli

    echo "Downloading additional genomes and annotations into ${MORE_GENOMES_DIR}..."
    for accession in "${ADDITIONAL_GENOMES[@]}"; do
        echo "Processing ${accession}..."
        if ! download_genome_with_datasets "$accession" "$MORE_GENOMES_DIR"; then
            echo "WARNING: Failed to download/prepare ${accession}, continuing with the next one." >&2
        fi
    done
fi

###############################################################################
# Prepare annotation datasets (Python scripts)
###############################################################################

# Original calls (UNCHANGED)
python3 minja_annotation/prepare_data.py \
    --genome_dir "$ANNOTATION_DIR" \
    --base_name gencode.v48.annotation.gtf \
    --fasta_name hg38.fa

python3 prepare_data.py \
    --genome_dir "$ANNOTATION_DIR" \
    --base_name GCF_009914755.1_T2T-CHM13v2.0.gff \
    --fasta_name T2T-CHM13v2.0_genomic.fna

# New: run prepare_data.py for each additional genome in its own folder (NEW)
for accession in "${ADDITIONAL_GENOMES[@]}"; do
    acc_dir="${MORE_GENOMES_DIR}/${accession}"
    fasta_name="genome.fna"

    if [ -f "${acc_dir}/annotation.gff3" ]; then
        gff_name="annotation.gff3"
    elif [ -f "${acc_dir}/annotation.gff" ]; then
        gff_name="annotation.gff"
    else
        echo "Skipping ${accession}: no annotation.gff3 or annotation.gff found in ${acc_dir}"
        continue
    fi

    if [ ! -f "${acc_dir}/${fasta_name}" ]; then
        echo "Skipping ${accession}: ${fasta_name} not found in ${acc_dir}"
        continue
    fi

    echo "Running prepare_data.py for ${accession}..."
    python3 prepare_data.py \
        --genome_dir "$acc_dir" \
        --base_name "$gff_name" \
        --fasta_name "$fasta_name"
done

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RAW_DIR="${SCRIPT_DIR}/data/raw"
PROCESSED_DIR="${SCRIPT_DIR}/data/processed"
BED_URL="https://www.encodeproject.org/files/ENCFF046XEZ/@@download/ENCFF046XEZ.bed.gz"
BED_GZ="${RAW_DIR}/ENCFF046XEZ.bed.gz"
BED_FILE="${RAW_DIR}/ENCFF046XEZ.bed"
UCSC_HG38_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
UCSC_FASTA_GZ="${RAW_DIR}/hg38.fa.gz"
UCSC_FASTA="${RAW_DIR}/hg38.fa"

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10000}"
VALID_CHROMS="${VALID_CHROMS:-chr21}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}"

if [[ ! -f "${UCSC_FASTA}" ]]; then
  echo "[1/4] Downloading UCSC hg38 FASTA ..."
  curl -L "${UCSC_HG38_URL}" -o "${UCSC_FASTA_GZ}"
  echo "[1/4] Decompressing hg38 FASTA ..."
  gzip -dc "${UCSC_FASTA_GZ}" > "${UCSC_FASTA}"
else
  echo "[1/4] Reusing existing UCSC hg38 FASTA: ${UCSC_FASTA}"
fi
FASTA_PATH="${UCSC_FASTA}"

echo "[2/4] Downloading ENCODE CTCF liver peaks ..."
curl -L "${BED_URL}" -o "${BED_GZ}"

echo "[3/4] Decompressing BED ..."
gzip -dc "${BED_GZ}" > "${BED_FILE}"

echo "[4/4] Building train/valid CSV files ..."
python "${SCRIPT_DIR}/prepare_ctcf_liver_dataset.py" \
  --fasta_path "${FASTA_PATH}" \
  --peaks_bed "${BED_FILE}" \
  --output_dir "${PROCESSED_DIR}" \
  --sequence_length "${SEQUENCE_LENGTH}" \
  --valid_chromosomes "${VALID_CHROMS}" \
  --max_samples "${MAX_SAMPLES}" \
  --seed "${SEED}"

echo "Done. Generated:"
echo "  ${PROCESSED_DIR}/train.csv"
echo "  ${PROCESSED_DIR}/valid.csv"

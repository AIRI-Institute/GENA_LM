#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODERNGENA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_DIR="${MODERNGENA_DIR}/data/raw"
PROCESSED_DIR="${MODERNGENA_DIR}/data/processed/token_regression"

UCSC_HG38_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
PEAKS_BED_URL="https://www.encodeproject.org/files/ENCFF046XEZ/@@download/ENCFF046XEZ.bed.gz"
SIGNAL_BW_URL="https://www.encodeproject.org/files/ENCFF184SOF/@@download/ENCFF184SOF.bigWig"

UCSC_FASTA_GZ="${RAW_DIR}/hg38.fa.gz"
UCSC_FASTA="${RAW_DIR}/hg38.fa"
PEAKS_BED_GZ="${RAW_DIR}/ENCFF046XEZ.bed.gz"
PEAKS_BED="${RAW_DIR}/ENCFF046XEZ.bed"
SIGNAL_BW="${RAW_DIR}/ENCFF184SOF.bigWig"

INTERVAL_LENGTH="${INTERVAL_LENGTH:-10000}"
SHIFT_BP="${SHIFT_BP:-5000}"
TEST_CHROMS="${TEST_CHROMS:-chr21}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}"

if [[ ! -f "${UCSC_FASTA}" ]]; then
  echo "[1/5] Downloading UCSC hg38 FASTA ..."
  curl -L "${UCSC_HG38_URL}" -o "${UCSC_FASTA_GZ}"
  echo "[1/5] Decompressing hg38 FASTA ..."
  gzip -dc "${UCSC_FASTA_GZ}" > "${UCSC_FASTA}"
else
  echo "[1/5] Reusing existing UCSC hg38 FASTA: ${UCSC_FASTA}"
fi

if [[ ! -f "${PEAKS_BED}" ]]; then
  echo "[2/5] Downloading ENCODE CTCF peak BED ..."
  curl -L "${PEAKS_BED_URL}" -o "${PEAKS_BED_GZ}"
  echo "[2/5] Decompressing BED ..."
  gzip -dc "${PEAKS_BED_GZ}" > "${PEAKS_BED}"
else
  echo "[2/5] Reusing existing CTCF BED: ${PEAKS_BED}"
fi

if [[ ! -f "${SIGNAL_BW}" ]]; then
  echo "[3/5] Downloading ENCODE CTCF signal bigWig ..."
  curl -L "${SIGNAL_BW_URL}" -o "${SIGNAL_BW}"
else
  echo "[3/5] Reusing existing signal bigWig: ${SIGNAL_BW}"
fi

echo "[4/5] Generating train/test intervals ..."
python "${SCRIPT_DIR}/prepare_ctcf_token_regression_dataset.py" \
  --fasta_path "${UCSC_FASTA}" \
  --peaks_bed "${PEAKS_BED}" \
  --output_dir "${PROCESSED_DIR}" \
  --interval_length "${INTERVAL_LENGTH}" \
  --shift_bp "${SHIFT_BP}" \
  --test_chromosomes "${TEST_CHROMS}" \
  --max_samples "${MAX_SAMPLES}" \
  --seed "${SEED}"

echo "[5/5] Done. Generated:"
echo "  ${PROCESSED_DIR}/train.csv"
echo "  ${PROCESSED_DIR}/test.csv"
echo "  Signal file (shared): ${SIGNAL_BW}"

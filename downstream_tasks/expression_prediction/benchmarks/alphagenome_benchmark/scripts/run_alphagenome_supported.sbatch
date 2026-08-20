#!/bin/bash
#SBATCH --job-name=alphagenome-supported
#SBATCH --output=alphagenome-supported-%j.out
#SBATCH --error=alphagenome-supported-%j.err
#SBATCH --partition=amd_256M
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
export GENA_HOME="${GENA_HOME:-$REPO}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$REPO}"
export DATA_ROOT="${DATA_ROOT:-$REPO/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$BENCHMARK_DIR"
mkdir -p "$BENCHMARK_ROOT/AlphaGenome/predictions"

"$PYTHON_BIN" "$SCRIPT_DIR/Alphagenome_prediction_benchmarking.py" \
  --config "$SCRIPT_DIR/config.yaml"

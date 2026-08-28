#!/bin/bash
#SBATCH --job-name=alphagenome-json
#SBATCH --partition=gpunode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=alphagenome_%j.out
#SBATCH --error=alphagenome_%j.err

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

export ALPHAGENOME_ONTOLOGY_JSON="${ALPHAGENOME_ONTOLOGY_JSON:-$BENCHMARK_DIR/data/alphagenome_track_to_qnorm_id_first.json}"

"$PYTHON_BIN" "$SCRIPT_DIR/Alphagenome_prediction_benchmarking_json.py" \
  --config "$SCRIPT_DIR/config_alphagenome_supported.yaml"

echo "Done"

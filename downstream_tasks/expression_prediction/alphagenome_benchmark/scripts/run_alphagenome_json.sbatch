#!/bin/bash
#SBATCH --job-name=alphagenome-json
#SBATCH --partition=gpunode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/biophysinf-alphagenome/logs/alphagenome_%j.out
#SBATCH --error=/scratch/biophysinf-alphagenome/logs/alphagenome_%j.err

set -euo pipefail

REPO=/home/biophysinf/DashaP/benchmarking/GENA_LM/GENA_LM
BENCHMARK_DIR=$REPO/downstream_tasks/expression_prediction/alphagenome_benchmark

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alphagenome

cd $REPO
mkdir -p /scratch/biophysinf-alphagenome/results /scratch/biophysinf-alphagenome/logs

export ALPHAGENOME_ONTOLOGY_JSON=${ALPHAGENOME_ONTOLOGY_JSON:-$BENCHMARK_DIR/data/alphagenome_track_to_qnorm_id_first.json}

python3 $BENCHMARK_DIR/scripts/Alphagenome_prediction_benchmarking_json.py \
  --config $BENCHMARK_DIR/scripts/config_alphagenome_supported.yaml

echo "Done"

#!/bin/bash
#SBATCH --job-name=alphagenome-supported
#SBATCH --output=/scratch/biophysinf-alphagenome/logs/slurm-%j.out
#SBATCH --error=/scratch/biophysinf-alphagenome/logs/slurm-%j.err
#SBATCH --partition=amd_256M
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

REPO=/home/biophysinf/DashaP/benchmarking/GENA_LM/GENA_LM
BENCHMARK_DIR=$REPO/downstream_tasks/expression_prediction/alphagenome_benchmark

cd $REPO
mkdir -p /scratch/biophysinf-alphagenome/results /scratch/biophysinf-alphagenome/logs

source /beegfs/datasets/home/biophysinf/miniconda3/etc/profile.d/conda.sh
conda activate alphagenome

python $BENCHMARK_DIR/scripts/Alphagenome_prediction_benchmarking.py \
  --config $BENCHMARK_DIR/scripts/config.yaml

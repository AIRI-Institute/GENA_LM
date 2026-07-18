#!/usr/bin/env bash
set -euo pipefail

# Submit the mammals evaluation array to the shared rnd partition.
# Each array task requests one GPU via slurm/mammals_eval.slurm.

cd "$(dirname "${BASH_SOURCE[0]}")"

export HF_HOME="${HF_HOME:-$HOME/.hf}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-0}"

# sbatch reads these variables as defaults for the array submission performed
# by submit_mammals_eval.sh. The child Slurm file intentionally has no
# partition/account/QoS directives of its own.
export SBATCH_PARTITION="${SBATCH_PARTITION:-rnd}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-airi}"
export SBATCH_QOS="${SBATCH_QOS:-airi-high}"

export CONDA_ENV="${CONDA_ENV:-$HOME/envs/gender}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

DATA_DIR="${DATA_DIR:-./data/mammals_data_contig_separated}"

echo "DATA_DIR=$DATA_DIR"
echo "CONDA_ENV=$CONDA_ENV"
echo "Slurm: partition=$SBATCH_PARTITION account=$SBATCH_ACCOUNT qos=$SBATCH_QOS"

DATA_DIR="$DATA_DIR" ./submit_mammals_eval.sh

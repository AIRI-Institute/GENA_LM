#!/usr/bin/env bash
# One-button launcher: finds the last checkpoint of every mammals finetuning
# run and submits a Slurm array job that evaluates all of them, one checkpoint
# per GPU, up to $MAX_CONCURRENT running at once.
#
# Usage:
#   DATA_DIR=/path/to/mammals_data_contig_separated ./submit_mammals_eval.sh
#
# Optional env vars:
#   RUNS_ROOT           default: runs
#   RUNS_PATTERN         fnmatch on the top-level dir under RUNS_ROOT, default:
#                        'mammals_contig_separated*' (the multi-species runs
#                        evaluate_all_mammals.sh's species lists are meant
#                        for). Pass '*' to include every run under RUNS_ROOT.
#   MAX_CONCURRENT      default: 8 (one per GPU on an 8xH200 node)
#   BATCH_SIZE          default: 8
#   N_PER_SAMPLE        default: 60_000
#   CONDA_ENV           default: gender-cls
#
# Review checkpoints_manifest.tsv before rerunning if you want to add/drop
# specific runs -- it's a plain TSV, safe to hand-edit and resubmit with
# `sbatch --array=0-<N-1>%<MAX_CONCURRENT> --export=ALL,DATA_DIR=...,MANIFEST=$(pwd)/checkpoints_manifest.tsv slurm/mammals_eval.slurm`.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

: "${DATA_DIR:?Set DATA_DIR=/path/to/mammals_data_contig_separated/}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
RUNS_PATTERN="${RUNS_PATTERN:-mammals_contig_separated*}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
MANIFEST="$(pwd)/checkpoints_manifest.tsv"

mkdir -p logs

echo "Scanning '$RUNS_ROOT' (pattern: '$RUNS_PATTERN') for last checkpoints..."
python3 find_last_checkpoints.py "$RUNS_ROOT" --pattern "$RUNS_PATTERN" > "$MANIFEST"

N=$(wc -l < "$MANIFEST")
if [ "$N" -eq 0 ]; then
    echo "No checkpoints found under '$RUNS_ROOT' matching '$RUNS_PATTERN' -- nothing to submit." >&2
    exit 1
fi

echo ""
echo "Found $N experiment(s):"
column -t -s $'\t' "$MANIFEST" | sed 's/^/  /'
echo ""
echo "Submitting Slurm array job (0-$((N - 1)), max $MAX_CONCURRENT concurrent)..."

sbatch \
    --array="0-$((N - 1))%${MAX_CONCURRENT}" \
    --export="ALL,MANIFEST=${MANIFEST},DATA_DIR=${DATA_DIR},BATCH_SIZE=${BATCH_SIZE:-8},N_PER_SAMPLE=${N_PER_SAMPLE:-60_000},CONDA_ENV=${CONDA_ENV:-gender-cls}" \
    slurm/mammals_eval.slurm

echo "Submitted. Track with: squeue --me ; logs land in logs/mammals_eval_<jobid>_<taskid>.{out,err}"

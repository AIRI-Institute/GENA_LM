#!/usr/bin/env bash

set -u
shopt -s nullglob

configs=(./configs/config_*.yaml)
pids=()

cleanup() {
    local exit_code="${1:-1}"

    # Prevent cleanup from being triggered recursively.
    trap - INT TERM

    echo "Terminating all running processes..." >&2

    # Negative PID targets the entire process group.
    for pid in "${pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done

    # Reap terminated processes.
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    exit "$exit_code"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM

if ((${#configs[@]} == 0)); then
    echo "No matching configuration files found." >&2
    exit 1
fi

for config in "${configs[@]}"; do
    echo "Starting: $config"

    # Start each command in a separate process group.
    setsid python ./run_generation.py --config "$config" &
    pids+=("$!")
done

remaining=${#pids[@]}

while ((remaining > 0)); do
    if wait -n; then
        ((remaining--))
    else
        exit_code=$?
        echo "A generation process failed with exit code $exit_code." >&2
        cleanup "$exit_code"
    fi
done

echo "All generation processes completed successfully."

#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash $0 /path/to/generation /path/to/api" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

for package_dir in "$@"; do
    if [[ ! -d "$package_dir" ]]; then
        echo "Error: package directory does not exist: $package_dir" >&2
        exit 1
    fi
done

wheelhouse_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
generation_dir="$(cd -- "$1" && pwd -P)"
api_dir="$(cd -- "$2" && pwd -P)"
python="${PYTHON:-python3}"

"$python" -m pip wheel --wheel-dir "$wheelhouse_dir" -- "$generation_dir" "$api_dir"

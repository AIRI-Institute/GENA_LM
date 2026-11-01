#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WHEELHOUSE="${ROOT}/wheelhouse"

REQUIREMENTS="${ROOT}/requirements.txt"
if [[ -f "${ROOT}/requirements.py" ]]; then
  REQUIREMENTS="${ROOT}/requirements.py"
fi

if [[ ! -d "${WHEELHOUSE}" ]]; then
  echo "Missing wheelhouse: ${WHEELHOUSE}" >&2
  exit 1
fi
if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "Missing requirements file: ${REQUIREMENTS}" >&2
  exit 1
fi

shopt -s nullglob
GENA_EXPRESSION_WHEELS=("${WHEELHOUSE}"/gena_expression-*.whl)
shopt -u nullglob
if [[ ${#GENA_EXPRESSION_WHEELS[@]} -ne 1 ]]; then
  echo "Expected exactly one gena_expression-*.whl in ${WHEELHOUSE}; found ${#GENA_EXPRESSION_WHEELS[@]}." >&2
  exit 1
fi

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

# Install the model/runtime dependency layer entirely from the bundled
# wheelhouse. All transitive wheels must also be present.
"${PYTHON_BIN}" -m pip install \
  --disable-pip-version-check \
  --no-index \
  --find-links "${WHEELHOUSE}" \
  --requirement "${REQUIREMENTS}"

# Force-reinstall the user-supplied API wheel so an older installation with the
# same distribution version cannot remain active.
"${PYTHON_BIN}" -m pip install \
  --disable-pip-version-check \
  --no-index \
  --no-deps \
  --upgrade \
  --force-reinstall \
  "${GENA_EXPRESSION_WHEELS[0]}"

# The generation package is deliberately shipped as source, not as a wheel.
# Disable dependency resolution and build isolation so this step cannot access
# the network and uses the build tools installed from the wheelhouse above.
"${PYTHON_BIN}" -m pip install \
  --disable-pip-version-check \
  --no-index \
  --no-deps \
  --no-build-isolation \
  --editable "${ROOT}"

"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata

import Bio
import flash_attn
import gena_expression
import hydra
import polars
import pysam
import safetensors
import tokenizers
import torch
import transformers
import yaml

import run_generation

print(
    f"gena-expression {metadata.version('gena-expression')}: "
    f"{gena_expression.__file__}"
)
print(
    f"gena-sequence-generation {metadata.version('gena-sequence-generation')}: "
    f"{run_generation.__file__}"
)
print(f"torch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
print("Offline installation smoke test passed.")
PY

# Build the wheelhouse

Build the wheels on an internet-connected machine with a Python version,
operating system, CPU architecture, and PyTorch/CUDA environment compatible
with the remote server. The examples use placeholder paths.

## Build the two packages

Run `scripts/build_wheels.sh` with the **generation** package directory first
and the **gena_expression API** package directory second. The API package lives
in `expression_prediction/api/`. Both arguments must point to installable
Python projects, not requirements files.

From the generation directory:

```bash
cd /path/to/generation
bash scripts/build_wheels.sh \
  /path/to/generation \
  /path/to/expression_prediction/api
```

The script uses `python3` by default. To select a virtual environment:

```bash
PYTHON=/path/to/venv/bin/python bash scripts/build_wheels.sh \
  /path/to/generation \
  /path/to/expression_prediction/api
```

The script writes wheels directly into `../wheelhouse` relative to its own
location: `generation/wheelhouse/` when the script is in `generation/scripts/`.
It creates that directory if needed, regardless of the current working
directory. This is the directory used by the offline installer.

The build includes both projects and their declared dependencies. It may need
compilers or other build tools when suitable prebuilt wheels are unavailable.

## Prepare the offline installer's requirements

The current checkout does not include `requirements.txt`, but
`scripts/install_offline.sh` requires it (or `requirements.py`, which takes
precedence if present). Create `requirements.txt` in the generation directory
before transferring the project.

The installer also imports Biopython and FlashAttention and installs the
generation source without build isolation. Its requirements therefore need
to cover those packages and the build tools in addition to the project
dependencies. A starting requirements file is:

```text
gena_expression>=0.1.0
polars>=1.38
PyYAML>=6.0
biopython
flash-attn
setuptools>=68
wheel
```

Use versions compatible with the target server, especially for FlashAttention
and the API's pinned PyTorch version. Preparing a FlashAttention wheel may
require a matching CUDA build environment and additional build prerequisites.
Include its wheel even if it is already installed on the build machine:
the installer imports it on the server.

With the same build interpreter used above, build the remaining requirements
into the wheelhouse. `--find-links` lets pip use the API wheel built in the
previous step:

```bash
/path/to/venv/bin/python -m pip wheel \
  --wheel-dir wheelhouse \
  --find-links wheelhouse \
  --requirement requirements.txt
```

If you used the default build interpreter, replace
`/path/to/venv/bin/python` with `python3`.

Existing wheels are not removed by either build command. The installer
requires exactly one `gena_expression-*.whl`; keep only the intended API
wheel if previous builds left multiple versions.

## Transfer and install

Transfer the generation source directory with `requirements.txt`,
`scripts/install_offline.sh`, and the populated `wheelhouse/` to the remote
server. Models, tokenizers, genome annotations, and experiment input files
must also be available there; they are not bundled into dependency wheels.

On the server:

```bash
cd /path/to/generation
python3 -m venv /path/to/venvs/generation
source /path/to/venvs/generation/bin/activate
PYTHON_BIN=python bash scripts/install_offline.sh
```

The installer uses the local wheelhouse with `--no-index`, installs the
generation source in editable mode, and runs dependency and import checks.
See the [package README](../README.md#remote-server-installation) for
configuration and launch instructions.


## Installation

The model is expected to run inside the base conda environment provided with
this repository. The environment file should be in the project root, next to the
`src/` directory.

Run all commands from the project root.

### 1. Create And Activate The Conda Environment

```bash
conda env create -f conda_env.yaml
conda activate <env-name>
```

Use the environment name declared at the top of `conda_env.yaml` in place of
`<env-name>`.

### 2. Install The Package

Install the API in editable mode so local source changes are immediately visible
to Python and Jupyter:

```bash
python -m pip install -e .
```

### 3. Install Model Runtime Dependencies

Install additional dependencies:

```bash
python -m pip install "transformers==4.55.2" "tokenizers==0.21.4"
python -m pip install "flash_attn==2.6.3" --no-build-isolation
```

### 4. Check The Installation

```bash
python -c "import gena_expression; print('gena_expression import OK')"
```
# GENA expression prediction API

- [Package reference](src/gena_expression/README.md) for conditions,
  annotated sequences, variants, and retention.
- [Model inference](src/gena_expression/inference/README.md) for pooling,
  batch execution, CUDA devices, and precision.
- [Stress testing](src/gena_expression/inference/README.md#inference-stress-test)
  for comparing predictions across execution settings.
- [DNA contexts and promoter sampling](src/gena_expression/dna/README.md).
- [Variant scoring](src/gena_expression/scoring/README.md) and
  [in-silico mutagenesis](src/gena_expression/ism/README.md).

## Installation

The API directory provides `conda_env.yaml` beside `src/`.

Run the installation commands from this API directory, which contains
`pyproject.toml`. If your environment already provides these dependencies,
activate it and install the package from this checkout.

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

Install the tokenizer and Transformers versions used by the model.

```bash
python -m pip install "transformers==4.55.2" "tokenizers==0.21.4"
```

For checkpoints whose default backend uses FlashAttention 2, install it in
the CUDA environment.

```bash
python -m pip install "flash_attn==2.6.3" --no-build-isolation
```

The opt-in FP32 mode uses PyTorch's SDPA math backend with compatible model
constructors. See [model inference](src/gena_expression/inference/README.md#precision).

### 4. Check The Installation

```bash
python -c "import gena_expression; print('gena_expression import OK')"
```

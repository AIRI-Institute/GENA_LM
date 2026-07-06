# Inference for Gene Expression Prediction

This notebook helps predict gene expression using two sources of information:

1. A DNA region around the gene.
2. A text description of the experiment.

## What you need

Before running the notebook, prepare:

- the `GENA_LM` repository;
- a model config file in `yaml` format;
- a model checkpoint file such as `pytorch_model.bin`;
- a genome fasta file, for example `hg38.fa`;
- a folder with cell descriptions in `json` format;
- a file with `forward` intervals;
- optionally, a file with `reverse` intervals.

## How to set up the environment

```bash
conda env create -f environment.yaml -n expression_flash
conda activate expression_flash
pip install -r ../../../requirements.txt

SOFT_DIR=$HOME/soft
mkdir -p $SOFT_DIR
cd $SOFT_DIR
git clone --branch feat/trainer_with_accelerate https://github.com/yurakuratov/t5-experiments.git
cd t5-experiments
sed -i 's/requirements-lm-tools.txt/requirements-lm-tools-accel.txt/' setup.py
python -m pip install -e .
pip install "transformers==4.55.2" "tokenizers==0.21.4"
pip install "flash_attn==2.6.3" --no-build-isolation
pip install ipykernel
python -m ipykernel install --user --name expression_flash --display-name "Python (expression_flash)"
pip install hydra-core --upgrade
```

## Where to run inference

The main file is `inference.ipynb`.

The notebook has a top cell called `# user-configurable variables`. This is where you should put your own paths and settings.

## What to fill in at the top of the notebook

### Required variables

- `GENA_HOME`  
  Path to the root of the `GENA_LM` repository.

- `EXPERIMENT_CONFIG`  
  Path to the `yaml` config file that contains model parameters.

- `CHECKPOINT_PATH`  
  Path to the model weights, usually a file like `pytorch_model.bin`.

- `JSON_DIR`  
  Path to the folder with cell descriptions in `json` format.

- `FORWARD_INTERVALS_PATH`  
  Path to the interval file for the `forward` strand.

- `GENOME_PATH`  
  Path to the reference genome in `fasta` format.

- `NUM_BEFORE`  
  How many tokens are taken before the TSS.

- `TOKEN_LEN_FOR_FETCH`  
  How many letters are used per token when extracting the sequence. It is best to keep the default value `15`. Changing it is not recommended unless you know exactly why you need it.

### Optional variables

- `INFERENCE_DIR`  
  Working directory for inference.  
  If `None`, the notebook uses `downstream_tasks/expression_prediction/inference_example` inside `GENA_HOME`.

- `REVERSE_INTERVALS_PATH`  
  Path to the interval file for the `reverse` strand.  
  If you only have `forward` intervals, leave it as `None`.

- `DNA_TOKENIZER`  
  You can explicitly specify the DNA tokenizer.  
  If `None`, it will be taken from the config.

- `TEXT_TOKENIZER`  
  You can explicitly specify the text tokenizer.  
  If `None`, it will be taken from the config.

- `DNA_MAX_SEQ_LEN`  
  Maximum DNA sequence length after tokenization.  
  If `None`, the value will be taken from the config.

- `TEXT_MAX_SEQ_LEN`  
  Maximum text description length after tokenization.  
  If `None`, the value will be taken from the config.

- `PREDICTION_MATRIX_CSV`  
  Name of the output `.csv` file with the `gene x cell type` table.

## Important rule about paths

If a path is absolute, it is used as is.

If a path is relative, it is interpreted relative to `INFERENCE_DIR`.

For example:

```python
JSON_DIR = "data/descriptions"
```

means that the folder will be searched for inside `INFERENCE_DIR`.

## What the interval file should look like

You can provide:

- only `forward` intervals;
- both `forward` and `reverse` intervals.

The file is read with `pandas`, so ordinary `csv` and `tsv` files are fine.

### Required columns

- `gene_id`  
  A unique gene identifier.

- `chromosome`  
  Chromosome name, for example `chr1`.

- `TSS`  
  Transcription start site coordinate.

- `TES`  
  Transcription end site coordinate.

### Optional column

- `gene_name`  
  A human-readable gene name. If it is missing, `gene_id` will be used instead.

### Important restrictions

- `gene_id` values must be unique.
- If you provide both `forward` and `reverse`, the same `gene_id` must not appear twice in the merged set.
- You do not need to include a `strand` column in these files: the notebook automatically treats the `forward` file as `"+"` and the `reverse` file as `"-"`.

### Example interval file

```csv
gene_id,gene_name,chromosome,TSS,TES
ENSG00000163631,ALB,chr4,73440227,73456844
ENSG00000206172,HBA1,chr16,176680,177522
```

## What the JSON descriptions should look like

The notebook expects a folder containing one or more `.json` files.

You can store:

- all `json` files in one folder;
- `json` files inside nested subfolders.

All `.json` files found there will be read automatically.

### What is required

Each file must be:

- valid `json`;
- non-empty;
- a dictionary-like object, meaning it should start with `{ ... }`.

### How it is turned into text

The code goes through all `key: value` pairs and builds a text like this:

```text
cell type is adipocyte. organism is Mus musculus. tissue is inguinal fat pad.
```

So there are no strictly required special fields such as `cell_type` or `organism`, but the file must be a meaningful dictionary.

### What is best in practice

The best option is to use a flat dictionary with simple fields such as:

- `cell_type`
- `organism`
- `genome`
- `tissue`
- `assay`

and any other useful description fields.

### Example of a good JSON file

```json
{
  "cell_type": "adipocyte",
  "organism": "Mus musculus",
  "tissue": "inguinal fat pad",
  "disease": "normal",
  "sex": "female",
  "assay": "10x 3' v3",
  "development_stage": "19-week-old stage"
}
```

### How experiment names are created

The experiment name is built from the file name.

For example:

- `adipocyte.json` -> `adipocyte`
- `mouse/fat/adipocyte.json` -> `mouse__fat__adipocyte`

## What happens when you run the notebook

After you fill in the top cell, the notebook does the following:

1. Loads the config and the model.
2. Loads the checkpoint.
3. Loads the tokenizers.
4. Reads all `json` descriptions.
5. Extracts DNA sequences from the genome using the interval files.
6. Tokenizes DNA in the same way as in the dataset.
7. Tokenizes the text descriptions.
8. Feeds both DNA and the description into the model.
9. Builds a prediction table and saves the matrix as `.csv`.

## What you get as output

The output includes:

- a table with columns `Cell Type`, `Gene`, `Predicted Expression`;
- a `gene x cell type` matrix;
- a `.csv` file whose name is set by `PREDICTION_MATRIX_CSV`.

## How caching works

To avoid tokenizing everything from scratch every time, the notebook saves intermediate files next to the inference run.

### DNA caches

Separate DNA caches are created:

- for `forward`;
- for `reverse`.

This is useful because if you first ran only `forward` and later added `reverse`, the `forward` part does not need to be recomputed.

The DNA cache hash depends on:

- the strand type: `forward` or `reverse`;
- the name, size, and modification time of the interval file;
- the name, size, and modification time of the genome file;
- the selected DNA tokenizer;
- `NUM_BEFORE`;
- `TOKEN_LEN_FOR_FETCH`.

### Description cache

Text descriptions also have a separate cache.

Its hash depends on:

- the name of the folder with `json` files;
- the number of `json` files;
- the total size of the `json` files;
- the selected text tokenizer;
- `TEXT_MAX_SEQ_LEN`.

## Common errors

### Error: file not found

Check:

- whether the path is correct;
- whether it is absolute or relative;
- whether `INFERENCE_DIR` is set correctly.

### Error: duplicate gene_id

This means the same `gene_id` appears more than once.  
Each `gene_id` must appear only once.

### Error: intervals file is missing required columns

Check that the file contains:

- `gene_id`
- `chromosome`
- `TSS`
- `TES`

### Error: empty JSON or invalid JSON

Check that:

- the file opens as normal JSON;
- it contains an object like `{ ... }`;
- it has at least one field.

## Short version: what a student should do

If you want the shortest possible version, the workflow is:

1. Open `inference.ipynb`.
2. Fill in the top cell with your paths.
3. Make sure you have a folder with `json` files and an interval file.
4. Run the cells from top to bottom.
5. Get the prediction table and the `.csv` matrix.

If something does not work, the problem is usually one of these three:

- an incorrect path;
- an invalid `json` format;
- wrong columns in the interval file.

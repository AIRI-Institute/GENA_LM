# Gene Finding Pipeline

Produces filtered transcript intervals (BED) from two evaluations on the same reference:

- **4-class**: TSS/PolyA probabilities → candidate transcript intervals.
- **6-class intragenic**: intragenic probabilities → filter candidates.

## Installation

1. Clone repos into project root `$PROOT` (set `PROOT` to your target path):

```bash
cd $PROOT
git clone https://github.com/AIRI-Institute/ModernBert/
git clone https://github.com/AIRI-Institute/GENA_LM/
cd GENA_LM
git checkout latest_annotation_branch
```

2. Create checkpoint dirs and place checkpoints (provided) there:

```bash
mkdir -p $PROOT/GENA_LM/runs/4class
mkdir -p $PROOT/GENA_LM/runs/6class
mkdir -p $PROOT/GENA_LM/runs/moderngena-base-pretrain-promoters_multi_v2_resume_ep30-ba90700/
```

3. Create conda env (from `GENA_LM` root):

```bash
cd $PROOT/GENA_LM/downstream_tasks/minja_annotation
conda env create -n bert24 -f bert24_env.yml
conda activate bert24
python -m pip install -r requirements.txt
pip install "flash_attn==2.6.3" --no-build-isolation
```

## FASTA input

Examples: produce `test.fa.bed` from `test.fa`.

- **Basic run:**

```bash
cd $PROOT/GENA_LM/downstream_tasks/minja_annotation/pipeline/
conda activate bert24
MODERNBERT_HOME="../../../../ModernBERT/" GENALM_HOME="../../../../GENA_LM/" python run_pipeline.py --fasta ../../../data/annotation/test.fa
```

- **Shift coordinates** (Shift output coordinates based on the fasta header so that you can visualize resulting file against reference genome:):

```bash
cd $PROOT/GENA_LM/downstream_tasks/minja_annotation/pipeline/
conda activate bert24
MODERNBERT_HOME="../../../../ModernBERT/" GENALM_HOME="../../../../GENA_LM/" python run_pipeline.py --fasta ../../../data/annotation/test.fa --shift UCSC
```

- **Custom temp dir:**

```bash
cd $PROOT/GENA_LM/downstream_tasks/minja_annotation/pipeline/
conda activate bert24
TMPDIR=/path/to/tmp TMP=/path/to/tmp TEMP=/path/to/tmp MODERNBERT_HOME="../../../../ModernBERT/" GENALM_HOME="../../../../GENA_LM/" python run_pipeline.py --fasta ../../../data/annotation/test.fa
```

Typical runtime ~1 min. Output path can be set with: `--bed_out`.

## Output

BED with identified transcripts (chr, start, end, strand). Strand: `+` → TSS in column 2; `-` → TSS in column 3. Note: some transcripts may share start or end.

```
chr6	31142439	31158196	-
chr6	31156646	31158196	-
chr6	31158537	31163191	+
```

## BigWig mode

Use precomputed BigWig predictions for a chromosome and run post-filtering + BED export only:

```bash
cd $PROOT/GENA_LM/downstream_tasks/minja_annotation/pipeline/
conda activate bert24
GENALM_HOME="../../../../GENA_LM/" python run_pipeline.py --bw4 "<path_to_4class_bw>" --bw6 "<path_to_6class_bw>" --chrom NC_060944.1 --bw_mode
```
For details on how to precompute BigWig files, see the [Evaluate](../README.md#evaluate)
0. Install [bedtools](https://anaconda.org/bioconda/bedtools)>2.30 and [samtools](https://anaconda.org/bioconda/samtools)

1. Download and process data (run from dnalm/data/yeasts directory):
```bash
bash download_and_process_yeast_data.sh
```

This will create fasta file `yeast_fasta_for_corpus_generation.fa.gz`, which can be further used as an input for [create_corpus.py](../../src/gena_lm/genome_tools/create_corpus.py):


```bash
python3 src/gena_lm/genome_tools/create_corpus.py --input-file data/yeasts/yeast_fasta_for_corpus_generation.fa.gz --output-dir data/yeasts/yeast_corpus/ --io-mode jsonl --min-len 10000
```

Data amount w/o augmentations, nucleotides:

train 1,345,434,353
test 82,338,353

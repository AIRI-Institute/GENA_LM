0. Install [bedtools](https://anaconda.org/bioconda/bedtools)>2.30, [samtools](https://anaconda.org/bioconda/samtools), and [NCBI datasets](https://www.ncbi.nlm.nih.gov/datasets/).

1. Download and process data (run from dnalm/data/athaliana directory):
```bash
bash download_and_process_athaliana_data.sh
```

This will create fasta file `athaliana_fasta_for_corpus_generation.fa.gz`, which can be further used as an input for [create_corpus.py](../../src/gena_lm/genome_tools/create_corpus.py):


```bash
python3 src/gena_lm/genome_tools/create_corpus.py --input-file data/athaliana/athaliana_fasta_for_corpus_generation.fa.gz --output-dir data/athaliana/athaliana_corpus/ --io-mode jsonl --min-len 10000 --n_augmentations 6
```

Data amount w/o augmentations, nucleotides:

train 4,147,986,974

test 346,219,192
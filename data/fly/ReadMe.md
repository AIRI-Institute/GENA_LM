0. Install [bedtools](https://anaconda.org/bioconda/bedtools)>2.30 and [samtools](https://anaconda.org/bioconda/samtools)

1. Download and process data (run from dnalm/data/fly directory):
```bash
bash download_and_process_fly_data.sh
```

This will create fasta file `fly_fasta_for_corpus_generation.fa.gz`, which can be further used as an input for [create_corpus.py](../../src/gena_lm/genome_tools/create_corpus.py). Note that we used `--n_augmentations 2` when generating corpus for _Drosophila_ data:

```
python3 src/gena_lm/genome_tools/create_corpus.py --input-file data/fly/fly_fasta_for_corpus_generation.fa.gz --output-dir data/fly/fly_corpus/ --io-mode jsonl --min-len 10000 --n_augmentations 2
```

Data amount w/o augmentations, nucleotides:

train 38,208,813,984

test 2,519,835,748

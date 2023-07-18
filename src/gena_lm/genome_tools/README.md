A single genome file can be processed with create_corpus.py

To download and process multiple genome datasets, there are following steps:
1. Download fasta files:
'cat ensemble_genomes_metadata.tsv | cut -f2 | xargs -L 1 wget -P "fasta/" --no-clobber'
2. Download and extract processed agp information into "contigs/" folder
3. Run
'bash create_multigenome_corpus.sh /path/to/full_ensembl_genomes_metadata /path/to/directory/with/fasta/ /path/to/directory/with/contigs/ /path/to/output/folder'
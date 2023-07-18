#!/bin/bash
# create_multigenome_corpus.sh metadata_table fasta_dir contigs_dir out_folder

set -e

metadata_table=$1
fasta_dir=$2
contigs_dir=$3
out_folder=$4

header=0
cat $metadata_table | while read line || [[ -n $line ]];
do
    if [[ header -eq 0 ]]
    then
		header=1;
		continue
	fi
	spname=$(echo $line | cut -d "," -f1)
	fastafile=$fasta_dir"/"$(basename $(echo $line | cut -d "," -f2))
	spoutdir=$out_folder"/"$spname
	contigs_split_file=$contigs_dir"/"${spname::-1}".breaks.csv"

	echo "Processing $spname"
	python create_corpus.py --input-file $fastafile --contigs-split-file $contigs_split_file --output-dir $spoutdir
done

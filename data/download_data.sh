#!/bin/bash
# Usage: download_data.sh <genome name>
# <genome name> can be human,mouse,ape,yeast,fly


download_datasets() {
    # detect which OS user is using and download corresponding datasets executable
    if [ "$(uname)" == "Darwin" ]; then
        # Do something under Mac OS X platform
        curl -o datasets 'https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/LATEST/mac/datasets'
        chmod +x datasets       
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Do something under GNU/Linux platform
        curl -o datasets 'https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/LATEST/linux-amd64/datasets'
        chmod +x datasets
    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
        # Do something under 32 bits Windows NT platform
        echo "MinGW is not supported yet"
    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
        # Do something under 64 bits Windows NT platform
        echo "MinGW is not supported yet"
    fi
    
}

# check if NCBI datasets is in available and download if needed
if [ ! -x "datasets" ]
then
    echo "NCBI datasets could not be found"
    echo "downloading datasets"
    download_datasets
else
    echo "Found datasets executable"
fi

GENOME=$1
if [[ $GENOME == "human" ]]; then
    #human_acc="GCF_000001405.39"
    human_acc="GCA_009914755.4" # switch to 2t2t-chm13 v2.0
    echo "Going to download human data with acc $human_acc"
    ./datasets download genome accession $human_acc --filename genomes/human_genome_dataset.zip
    unzip genomes/human_genome_dataset.zip
elif [[ $GENOME == "mouse" ]]; then
    echo "genome not supported"
elif [[ $GENOME == "ape" ]]; then
    echo "genome not supported"
elif [[ $GENOME == "yeast" ]]; then
    echo "genome not supported"
elif [[ $GENOME == "fly" ]]; then
    echo "genome not supported"
elif [[ -z $GENOME ]]; then
    echo "Please select the genome - human,mouse,ape,yeast,fly"
fi
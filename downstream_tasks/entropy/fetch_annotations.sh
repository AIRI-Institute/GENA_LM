mkdir -p data/annotations
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/nestedRepeats.txt.gz -O data/annotations/nestedRepeats.txt.gz
gunzip data/annotations/nestedRepeats.txt.gz

wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/simpleRepeat.txt.gz -O data/annotations/simpleRepeat.txt.gz
gunzip data/annotations/simpleRepeat.txt.gz

python3 preprocess_annotations.py
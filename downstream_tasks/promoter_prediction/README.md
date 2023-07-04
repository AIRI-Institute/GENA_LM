# Promoter prediction

We compared performance of GENA-LM models to 
1. DeePromoter https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6460014/
2. BigBird https://papers.nips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html
3. DNABERT https://pubmed.ncbi.nlm.nih.gov/33538820/

## Dataset Preparation

### Step 1. Download data
EPDNew https://epd.epfl.ch/EPDnew_select.php database is used to select human promoters (hg38)
Four different sequence lengths are used:
1) Length 300. From -249 to 50. Results in a file hg38_len_300.fa.txt
2) Length 2000. From -1000 to 999. Results in a file hg38_len_2000.fa.txt
3) Length 8000 (like in BigBird paper). From -5000 to  2999. Results in a file hg38_len_8000.fa.txt
4) Length 16000. From -8000 to 7999. Results in a file hg38_len_16000.fa.txt

### Step 2. Create a dataset
Run the script `dataset_generator.py`` with fasta files obtained in previous step.
```
>> python dataset_generator.py
hg38_len_300.fa.txt
```
The script treats promoter sequences as positive targets and generates negative samples, following the same procedure as in DeePromoter paper.
Results in:
```
hg38_promoters_len_300_dataset.csv
```
### Step 3. Split to 5 folds
Run the dataset_fold_split.py script with csv files obtained from dataset generator
```
>> python dataset_fold_split.py
hg38_promoters_len_300_dataset.csv
```
Results in five csv files named from fold_1.csv to fold_5.csv and corresponding train/valid/test splits.

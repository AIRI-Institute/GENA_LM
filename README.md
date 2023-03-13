# GENA-LM

GENA-LM is a transformer masked language model trained on human DNA sequence.

Differences between GENA-LM and DNABERT:
- BPE tokenization instead of k-mers;
- input sequence size is about 3000 nucleotides (512 BPE tokens) compared to 510 nucleotides of DNABERT
- pre-training on T2T vs. GRCh38.p13 human genome assembly.


## Examples
### How to load the model to fine-tune it on classification task
```python
from src.gena_lm.modeling_bert import BertForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base')
```

## Model description
GENA-LM model is trained in a masked language model (MLM) fashion, following the methods proposed in the BigBird paper by masking 85% of tokens. Model config for `gena-lm-bert-base` is similar to the bert-base:

- 512 Maximum sequence length
- 12 Layers, 12 Attention heads
- 768 Hidden size
- 32k Vocabulary size

We pre-trained `gena-lm-bert-base` using the latest T2T human genome assembly (https://www.ncbi.nlm.nih.gov/assembly/GCA_009914755.3/). Pre-training was performed for 500,000 iterations with the same parameters as in BigBird, except sequence length was equal to 512 tokens and we used pre-layer normalization in Transformer.


### Download and preprocess data
In order to download human genome please run the following script:
```
./download_data.sh human
```

For preprocessing, execute the following script:

```
python src/gena_lm/genome_tools/create_corpus.py --input_file data/ncbi_dataset/data/GCA_009914755.4/GCA_009914755.4_T2T-CHM13v2.0_genomic.fna --output_dir data/processed/human/
```



## Downstream tasks
Currently, gena-lm-bert-base model has been finetuned  and tested on promoter prediction task.  Its' performance is comparable to previous SOTA results. We plan to fine-tune and make available models for other downstream tasks in the near future.

### Promoter Prediction
Performance of gena-lm-bert-base is compared to 
1. DeePromoter https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6460014/
2. BigBird https://papers.nips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html

#### Dataset Preparation

##### Step 1. Download data
EPDNew https://epd.epfl.ch/EPDnew_select.php database is used to select human promoters (hg38)
Three different sequence lengths are used:
1) Length 300. From -249 to 50. Results in a file hg38_len_300.fa.txt
2) Length 2000. From -1000 to 999. Results in a file hg38_len_2000.fa.txt
3) Length 16000. From -8000 to 7999. Results in a file hg38_len_16000.fa.txt

##### Step 2. Create a dataset
Run the script dataset_generator.py with fasta files obtained in previous step.
```
>> python dataset_generator.py
hg38_len_300.fa.txt
```
The script treats promoter sequences as positive targets and generates negative samples, following the same procedure as in DeePromoter paper.
Results in:
```
hg38_promoters_len_300_dataset.csv
```
##### Step 3. Split to 5 folds
Run the dataset_fold_split.py script with csv files obtained from dataset generator
```
>> python dataset_fold_split.py
hg38_promoters_len_300_dataset.csv
```
Results in five csv files named from fold_1.csv to fold_5.csv that need to be saved into a specified directory.

### Fine-tuning GENA-LM on our data and scoring
After fine-tuning gena-lm-bert-base on promoter prediction dataset, following results were achieved: 

| model                    | seq_len (bp) | F1    |
|--------------------------|--------------|-------|
| DeePromoter              | 300          | 95.60 |
| GENA-LM bert-base (ours) | 2000         | 95.72 |
| BigBird                  | 16000        | 99.90 |

We can conclude that our model achieves comparable performance to the previously published results for promoter prediction task.

## Installation
For models with sparse attention FP16 support and DeepSpeed is needed.
### APEX for FP16
Install APEX https://github.com/NVIDIA/apex#quick-start
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### DeepSpeed
DeepSpeed installation is needed to work with SparseAttention versions of language models. DeepSpeed Sparse attention supports only GPUs with compute compatibility >= 7 (V100, T4, A100), CUDA 10.1, 10.2, 11.0, or 11.1 and runs only in FP16 mode (as of DeepSpeed 0.6.0).
```bash
pip install triton==1.0.0
DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.6.0 --global-option="build_ext" --global-option="-j8" --no-cache
```
and check installation with
```bash
ds_report
```

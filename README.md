# DNALM

The project is dedicated to the development of language model capable to work with DNA sequences.

DeepSpeed installation is needed to work with SparseAttention versions of language models.

## Examples
### How to load the model to fine-tune it on classification task
```python
from src.dnalm.modeling_bert import BertForSequenceClassification, BertConfig
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./data/tokenizers/human/BPE_32k/')
config = BertConfig.from_pretrained('./data/configs/L12-H768-A12-V32k-preln.json')
model = BertForSequenceClassification(config=config)

# load pre-trained weights if you have them
import torch
ckpt_path = 'PATH_TO_CKPT'
checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
```

## Installation
### APEX
Install APEX https://github.com/NVIDIA/apex#quick-start
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### DeepSpeed
DeepSpeed Sparse attention supports only GPUs with compute compatibility >= 7 (V100, T4, A100), CUDA 10.1, 10.2, 11.0, or 11.1 and runs only in FP16 mode (as of DeepSpeed 0.6.0).
```bash
pip install triton==1.0.0
DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.6.0 --global-option="build_ext" --global-option="-j8" --no-cache
```
and check installation with
```bash
ds_report
```

## Downstream tasks
DNALM is tested on a multiple downstream tasks. Its' performance is compared to previous SOTA results

### Promoter Prediction
Performance of DNALM is compared to 
1. DeePromoter https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6460014/
2. BigBird https://papers.nips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html

#### Dataset Preparation

##### Step 1. Download data
EPDNew https://epd.epfl.ch/get_promoters.php database is used to select human promoters (hg38)
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
Results in five csv files named from fold_1.csv to fold_5.csv that need to be saved into a specified directory

### Fine-tuning DNALM on our data and scoring

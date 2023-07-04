# GENA-LM

GENA-LM is a family of Open-Source Foundational Models for Long DNA Sequences.

GENA-LM models are transformer masked language models trained on human DNA sequence.

Key features of our GENA-LM models:
- BPE tokenization instead of k-mers (DNABERT, Nucleotide Transformer)
- max input sequence size ranges from 4.5k to 36k bp, compared to 512bp in DNABERT and 1000bp in Nucleotide Transformer
- pre-training on the latest [T2T](https://www.ncbi.nlm.nih.gov/assembly/GCF_009914755.1/) human genome assembly vs GRCh38/hg38


## Pre-trained models

| Model                                                                                            | Architecture                         | Max SeqLen, tokens (bp) | Params | Tokenizer data              | Training data               |
| ------------------------------------------------------------------------------------------------ | ------------------------------------ | ----------------------- | ------ | --------------------------- | --------------------------- |
| [bert-base](https://huggingface.co/AIRI-Institute/gena-lm-bert-base)                             | BERT-12L                             | 512(4500)               | 110M   | T2T split v1                | T2T split v1                |
| [bert-base-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bert-base-t2t)                     | BERT-12L                             | 512(4500)               | 110M   | T2T+1000G SNPs+Multispecies | T2T+1000G SNPs              |
| [bert-base-lastln-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bert-base-lastln-t2t)       | BERT-12L                             | 512(4500)               | 110M   | T2T+1000G SNPs+Multispecies | T2T+1000G SNPs              |
| [bert-base-t2t-multi](https://huggingface.co/AIRI-Institute/gena-lm-bert-base-t2t-multi)         | BERT-12L                             | 512(4500)               | 110M   | T2T+1000G SNPs+Multispecies | T2T+1000G SNPs+Multispecies |
| [bert-large-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bert-large-t2t)                   | BERT-24L                             | 512(4500)               | 336M   | T2T+1000G SNPs+Multispecies | T2T+1000G SNPs              |
| [bigbird-base-sparse](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-sparse)         | BERT-12L, DeepSpeed Sparse Ops, RoPE | 4096(36000)             | 110M   | T2T split v1                | T2T split v1                |
| [bigbird-base-sparse-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-sparse-t2t) | BERT-12L, DeepSpeed Sparse Ops, RoPE | 4096(36000)             | 110M   | T2T+1000G SNPs+Multispecies | T2T+1000G SNPs              |
| [bigbird-base-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-t2t)               | BERT-12L, HF BigBird                 | 4096(36000)             | 110M   | T2T+1000G SNPs+Multispecies | T2T+1000G SNPs              |

T2T split v1 refers to preliminary models with a non-augmented T2T human genome assembly split. BERT-based models employ [Pre-Layer Normalization](https://arxiv.org/abs/2002.04745) and lastln explicitly denotes that layer normalization is also applied to the final layer. RoPE indicates the use of [rotary position embeddings](https://arxiv.org/abs/2104.09864) in place of BERT-like absolute positional embeddings.

For our first models (gena-lm-bert-base and gena-lm-bigbird-base-sparse) we hold out human chromosomes 22 and Y (CP068256.2 and CP086569.2) as the test dataset for the masked language modeling task. For all other models, we hold out human chromosomes 7 and 10 (CP068271.2 and CP068268.2); these models have the suffix "t2t" in their names. Other data was used for training. Human-only models were trained on pre-processed Human T2T v2 genome assembly and its 1000-genome SNP augmentations making in a total of ≈ 480 x 10^9 base pairs. Multispecies models were trained on human-only and multispecies data making in a total of ≈ 1072×109 base pairs.


## Examples
### How to load pre-trained GENA-LM for Masked Language Modeling
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t')
model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', trust_remote_code=True)

```

### How to load pre-trained GENA-LM to fine-tune it on classification task


Get model class from GENA-LM repository:
```bash
git clone https://github.com/AIRI-Institute/GENA_LM.git
```

```python
from GENA_LM.src.gena_lm.modeling_bert import BertForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base')
```
or you can just download [modeling_bert.py](https://github.com/AIRI-Institute/GENA_LM/tree/main/src/gena_lm) and put it close to your code.

OR you can get model class from HuggingFace AutoModel:
```python
model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', trust_remote_code=True)
gena_module_name = model.__class__.__module__
print(gena_module_name)
import importlib
# available class names:
# - BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
# - BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
# - BertForQuestionAnswering
# check https://huggingface.co/docs/transformers/model_doc/bert
cls = getattr(importlib.import_module(gena_module_name), 'BertForSequenceClassification')
print(cls)
model = cls.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', num_labels=2)
```


## Citation
```
@article {GENA_LM,
	author = {Veniamin Fishman and Yuri Kuratov and Maxim Petrov and Aleksei Shmelev and Denis Shepelin and Nikolay Chekanov and Olga Kardymon and Mikhail Burtsev},
	title = {GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences},
	elocation-id = {2023.06.12.544594},
	year = {2023},
	doi = {10.1101/2023.06.12.544594},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.12.544594},
	eprint = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.12.544594.full.pdf},
	journal = {bioRxiv}
}
```

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
Four different sequence lengths are used:
1) Length 300. From -249 to 50. Results in a file hg38_len_300.fa.txt
2) Length 2000. From -1000 to 999. Results in a file hg38_len_2000.fa.txt
3) Length 8000 (like in BigBird paper). From -5000 to  2999. Results in a file hg38_len_8000.fa.txt
4) Length 16000. From -8000 to 7999. Results in a file hg38_len_16000.fa.txt

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
Results in five csv files named from fold_1.csv to fold_5.csv and corresponding train/valid/test splits.

### Fine-tuning GENA-LM on our data and scoring
After fine-tuning gena-lm-bert-base on promoter prediction dataset, following results were achieved: 

| model                    | seq_len (bp) | F1    |
| ------------------------ | ------------ | ----- |
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

### Finetuning with lm-experiments-tools
We use Trainer and multi-gpu training from [lm-experiments-tools](https://github.com/yurakuratov/t5-experiments) repository as the basis for our finetuning scripts. However, you can use  HF Trainer, PyTorch Lighting, or Accelerate and PyTorch with custom training loops instead.

Install lm-experiments-tools according to https://github.com/yurakuratov/t5-experiments#install-only-lm_experiments_tools:
```
git clone https://github.com/yurakuratov/t5-experiments
cd t5-experiments
pip install -e .
```
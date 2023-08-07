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

For our first models (`gena-lm-bert-base` and `gena-lm-bigbird-base-sparse`) we hold out human chromosomes 22 and Y (CP068256.2 and CP086569.2) as the test dataset for the masked language modeling task. For all other models, we hold out human chromosomes 7 and 10 (CP068271.2 and CP068268.2); these models have the suffix "t2t" in their names. Other data was used for training. Human-only models were trained on pre-processed Human T2T v2 genome assembly and its 1000-genome SNP augmentations making in a total of ≈ 480 x 10^9 base pairs. Multispecies models were trained on human-only and multispecies data making in a total of ≈ 1072 x 10^9 base pairs.

## Pre-trained models on downstream tasks
| Model                  | Task        | Task seq len | Metric             | HF branch name       |
| ---------------------- | ----------- | ------------ | ------------------ | -------------------- |
| gena-lm-bert-base-t2t  | promoters   | 300bp        | 74.56+-0.36 F1     | promoters_300_run_1  |
| gena-lm-bert-large-t2t | promoters   | 300bp        | 76.44+-0.16 F1     | promoters_300_run_1  |
| gena-lm-bert-large-t2t | promoters   | 2000bp       | 93.70+-0.44 F1     | promoters_2000_run_1 |
| gena-lm-bert-base-t2t  | splice cite | 15000bp      | 92.63+-0.09 PR AUC | spliceai_run_1       |
| gena-lm-bert-large-t2t | splice cite | 15000bp      | 93.59+-0.11 PR AUC | spliceai_run_1       |

To get a pre-trained model on a downstream task, replace `model_name` and `branch_name` with values from the table. The metrics in the table are averaged over multiple runs. Therefore, the values for each checkpoint may differ from those reported here.


```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')
model = AutoModel.from_pretrained(f'AIRI-Institute/{model_name}', revision=branch_name, trust_remote_code=True)
```

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

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t')
model = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t')
```
or you can just download [modeling_bert.py](./src/gena_lm/modeling_bert.py) and put it close to your code.

OR you can get model class from HuggingFace AutoModel:
```python
from transformers import AutoTokenizer, AutoModel
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

GENA-LM `bigbird-base-t2t` model uses the HuggingFace BigBird implementation. Therefore, default classes from the Transformers library could be used:
```python
from transformers import AutoTokenizer, BigBirdForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bigbird-base-t2t')
model = BigBirdForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bigbird-base-t2t')
```

### Notebooks
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIRI-Institute/GENA_LM/blob/main/notebooks/GENA_sequence_classification_example.ipynb) Sequence classification with GENA-LM and Huggingface Transformers

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

## Downstream tasks
Downstream tasks for model evaluation encompass the prediction of promoter and enhancer activity, splicing sites, chromatin profiles, and polyadenylation site strength.
Check `downstream_tasks` folder for code and data preprocessing scripts we used:
- [Promoters prediction](./downstream_tasks/promoter_prediction/)
- [Splice cite prediction (SpliceAI)](./downstream_tasks/SpliceAI/)
- [Drosophila enhancers prediction (DeepSTARR)](./downstream_tasks/DeepSTARR/)
- [Chromatin profiling (DeepSea)](./downstream_tasks/DeepSea/)
- Polyadenylation sites prediction (APARENT)

## Pre-training data

### Download and preprocess data
In order to download human genome please run the following script:
```
./download_data.sh human
```

For preprocessing, execute the following script:
```
python src/gena_lm/genome_tools/create_corpus.py --input_file data/ncbi_dataset/data/GCA_009914755.4/GCA_009914755.4_T2T-CHM13v2.0_genomic.fna --output_dir data/processed/human/
```

## Installation
For models with sparse attention (`gena-lm-bigbird-base-sparse`, `gena-lm-bigbird-base-sparse-t2t`) FP16 support and DeepSpeed is needed.
### APEX for FP16
Install APEX https://github.com/NVIDIA/apex#quick-start
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### DeepSpeed for Sparse Ops
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
We use Trainer and multi-gpu training from [lm-experiments-tools](https://github.com/yurakuratov/t5-experiments) repository as the basis for our finetuning scripts. However, you can use  HF Transformers Trainer, PyTorch Lightning, or Accelerate and PyTorch with custom training loops instead.

Install lm-experiments-tools according to https://github.com/yurakuratov/t5-experiments#install-only-lm_experiments_tools:
```
git clone https://github.com/yurakuratov/t5-experiments
cd t5-experiments
pip install -e .
```
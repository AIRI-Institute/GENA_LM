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


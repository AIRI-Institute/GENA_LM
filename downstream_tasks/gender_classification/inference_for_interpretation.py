import importlib
from tqdm.auto import tqdm
import os
import pickle

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_model
from model import GenderChunkedClassifier
from eval_utils.inference_utils import infer

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t')
model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', trust_remote_code=True)
model_module = importlib.import_module(model.__class__.__module__)
cls = getattr(model_module, 'BertModel')
model = cls.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', add_pooling_layer=False)

gender_model = GenderChunkedClassifier(model)


load_model(gender_model,
           
           './runs/human_and_mouse_fixed_chromosome_ratios_16x3072_bs_128_lr_1e-05_chrY_with_SNPs/run_1/checkpoint-99500/model.safetensors',
           )

n_chunks = 16
chunk_size = 3072
seed = 142
bs = 2
split_name='test'
N = 300_000

chrX_ratio = None
chrY_ratio = None
force_sampling_from_y=True
force_label = [0]

sample_ids_labels, sample_ids_probs, sample_ids_sampled_chromosomes, sample_ids_attn_scores = infer(gender_model, tokenizer, split_name,
        n_chunks=n_chunks, chunk_size=chunk_size, seed=142, batch_size=bs, chrX_ratio=chrX_ratio, chrY_ratio=chrY_ratio, 
        force_sampling_from_y=force_sampling_from_y, force_label=force_label, N=N)

# # # saving 
pickle.dump({'sample_ids_probs': sample_ids_probs, 'sample_ids_labels': sample_ids_labels,
        'sample_ids_sampled_chromosomes': sample_ids_sampled_chromosomes, "sample_ids_attn_scores": sample_ids_attn_scores},
            open(f'{split_name}_Y_ratio_{chrY_ratio}_X_ratio_{chrX_ratio}_{N}_human{seed}_forced_y_sampling_{force_sampling_from_y}_with_chr_positions_and_attn_scores.pckl', 'wb'))
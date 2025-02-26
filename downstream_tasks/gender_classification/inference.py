import importlib
from tqdm.auto import tqdm
import os
import pickle

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np 
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import GenderChunkedClassifier
from gender_dataset import MultiSpeciesGenderDataChunkedDataset, worker_init_fn, collate_fn

from safetensors.torch import load_model


tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t')
model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', trust_remote_code=True)
model_module = importlib.import_module(model.__class__.__module__)
cls = getattr(model_module, 'BertModel')
model = cls.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', add_pooling_layer=False)

gender_model = GenderChunkedClassifier(model)


load_model(gender_model,
           
           '/home/jovyan/chepurova/dnalm/downstream_tasks/gender_classification/runs/human_and_mouse_fixed_16x3072_bs_128_lr_1e-05_chrY_with_SNPs/run_1/checkpoint-94500/model.safetensors',
           )

gender_model = gender_model.cuda()          

def preprocess_collate_fn(samples):
    batch = collate_fn(samples)

    batch['chunks'] = np.array(batch['chunks'])
    shape = batch['chunks'].shape
    batch['chunks'] = list(batch['chunks'].flatten())

    tokenized_batch = tokenizer(batch['chunks'], padding='longest', max_length=512,
                                truncation=True, return_tensors='pt')

    for k in tokenized_batch:
        tokenized_batch[k] = tokenized_batch[k].reshape(*shape, -1)

    batch['labels'] = torch.Tensor(batch['labels'])

    return {
        'input_ids': tokenized_batch['input_ids'],
        'attention_mask': tokenized_batch['attention_mask'],
        'labels': batch['labels'],
        'sample_ids': batch['sample_id'],
        'has_y_chr_sampled': batch['has_y_chr_sampled'],
        'has_x_chr_sampled': batch['has_x_chr_sampled'],
        'sampled_chromosomes': batch['sampled_chromosomes']
        }


n_chunks = 16
chunk_size = 3072
seed = 142
bs = 64

params = [('valid', 10**(-5), None), ('test', 10**(-5), None), ('valid', None, 10**(-5)), ('test', None, 10**(-5)), \
        ('valid', None, 0), ('test', None, 0), ('valid', 0, None), ('test', 0, None)]

for param in params:

    split_name, chrX_ratio, chrY_ratio = param
    print(f"Inferring for {split_name}, chrX_ratio: {chrX_ratio}, chrY_ratio: {chrY_ratio}")

    valid_dataset = MultiSpeciesGenderDataChunkedDataset(split_name=split_name, n_chunks=n_chunks,
                                            chunk_size=chunk_size,
                                            force_sampling_from_y=False,
                                            chrY_ratio=chrY_ratio,
                                            chrX_ratio=chrX_ratio,
                                            seed=seed+1)


    valid_dataloader = DataLoader(valid_dataset, batch_size=bs, collate_fn=preprocess_collate_fn,
                            num_workers=2, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    dataloader = valid_dataloader

    N = 100000 // bs

    labels = []
    probs = []
    sample_ids_list = []
    contain_y_list = []
    sampled_chromosomes_list = []

    for b in tqdm(dataloader, total=N):

        # assert all(('chrX' not in chr) and ('chrY' not in chr) for chr in b['sampled_chromosomes'])
        # assert all('chrX' not in chr for chr in b['sampled_chromosomes'])
        # print(b['sampled_chromosomes'])

        for k in b:
            # put numerical features on gpu
            if k not in ['sample_ids', 'has_y_chr_sampled', 'has_x_chr_sampled', 'sampled_chromosomes']:
                b[k] = b[k].cuda()


        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                outputs = gender_model(b['input_ids'], b['attention_mask'])
                labels += list(b['labels'].cpu().numpy())
                probs += list(outputs['predictions'].float().cpu().numpy())
                sample_ids_list += b['sample_ids']
                sampled_chromosomes_list += b['sampled_chromosomes']
        N -= 1
        if N <= 0:
            break

    labels = np.array(labels)
    probs = np.array(probs)

    sample_ids_labels = {sample_id: label for sample_id, label in zip(sample_ids_list, labels)}

    sample_ids_sampled_chromosomes = {}

    sample_ids_probs = {}

    # aggregating sampled chromosomes for each sample 
    for i, chrs in zip(sample_ids_list, sampled_chromosomes_list):
        if i in sample_ids_sampled_chromosomes:
            sample_ids_sampled_chromosomes[i] += chrs
        else:
            sample_ids_sampled_chromosomes[i] = chrs

    # aggregating probabilities for each sample 
    for i, p in zip(sample_ids_list, probs):
        if i in sample_ids_probs:
            sample_ids_probs[i] += [p]
        else:
            sample_ids_probs[i] = [p]

    # converting prob arrays to numpy
    for k in sample_ids_probs:
        sample_ids_probs[k] = np.array(sample_ids_probs[k])

    for k in sample_ids_probs:
        probs = sample_ids_probs[k]
        print(f'{k}: [{sample_ids_labels[k]}] {(probs>0.5).sum() / len(probs):.2f}')

    for k in sample_ids_probs:
        probs = sample_ids_probs[k]
        print(f'{k:20s}: [{sample_ids_labels[k]}] {(probs>5e-1).sum()}/{len(probs)} = {(probs>5e-1).sum() / len(probs):.3f}')


    pickle.dump({'sample_ids_probs': sample_ids_probs, 'sample_ids_labels': sample_ids_labels, 'sample_ids_sampled_chromosomes': sample_ids_sampled_chromosomes},
                open(f'{split_name}_Y_ratio_{chrY_ratio}_X_ratio_{chrX_ratio}_100k_human_mouse_Ysnps{seed}.pckl', 'wb'))
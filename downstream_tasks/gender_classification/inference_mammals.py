import argparse
import importlib
import logging
import os
import pickle
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_model

from mammals_gender_dataset import (MultiSpeciesGenderDataChunkedDataset, collate_fn,
                          worker_init_fn)
from model import GenderChunkedClassifier
from eval_utils.inference_utils import find_threshold_for_N, calculate_metrics
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Gender classification inference script')
    # Model and data paths
    parser.add_argument('--model_path', type=str, 
                    default='runs/human_mouse_contigs_16x3072_bs_128_lr_1e-05_chrY_chrY_ratio_0.5/run_1/checkpoint-153250/model.safetensors',
    )
    parser.add_argument('--data_dir', type=str,
                    #   default='/disk/10tb/home/chepurova/human_data_contigs_separated',
                      default='/disk/10tb/home/chepurova/mouse_data_contigs_separated_2',
                      help='Directory containing the data files')
    
    # Model configuration
    parser.add_argument('--n_chunks', type=int, default=16,
                      help='Number of chunks per sample')
    parser.add_argument('--chunk_size', type=int, default=3072,
                      help='Size of each chunk')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for inference')
    parser.add_argument('--n_per_sample', type=int, default=30_000,
                      help='Number of predictions per sample')
    parser.add_argument('--force_sampling_from_y', type=bool, default=False, help='Forcing sampling from y chromosome')
    
    # Training configuration
    parser.add_argument('--seed', type=int, default=142,
                      help='Random seed')
    parser.add_argument('--num_workers', type=int, default=2,
                      help='Number of data loading workers')
    
    # Data configuration
    parser.add_argument('--split', type=str, default='test',
                      choices=['train', 'valid', 'test', 'human_test'],
                      help='Data split to use')
    parser.add_argument('--chrX_ratio', type=float, default=None,
                      help='Ratio for chromosome X sampling')
    parser.add_argument('--chrY_ratio', type=float, default=None,
                      help='Ratio for chromosome Y sampling')
    
    parser.add_argument('--force_species', type=str, default=None,
                      help='Species for evaluation')
    
    parser.add_argument('--output_file_prefix', type=str, default='',
                        help='Output filename prefix')
    
    parser.add_argument('--save_probs', action='store_true',
                        help='Save probabilities')

    parser.add_argument('--metrics_output_file_prefix', type=str, default='evaluation_results.jsonl',
                        help='Metrics output file prefix')
    
    return parser.parse_args()


def preprocess_collate_fn(samples, tokenizer):
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
        'sample_ids': batch['sample_ids'],
        'species': batch['species'],
        'sampled_chromosomes': batch['sampled_chromosomes']
    }


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t')
    base_model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', trust_remote_code=True)
    model_module = importlib.import_module(base_model.__class__.__module__)
    cls = getattr(model_module, 'BertModel')
    model = cls.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', add_pooling_layer=False)
    return model, tokenizer


def process_batches(dataloader, model, N, species):
    labels = []
    probs = []
    sampled_chromosomes = []
    attn_scores = []

    for batch in tqdm(dataloader, total=N):
        assert all([id_ == species for id_ in batch['species']])
        
        # Move numerical features to GPU
        for k in batch:
            if k not in ['sample_ids', 'species', 'sampled_chromosomes']:
                batch[k] = batch[k].cuda()

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                outputs = model(batch['input_ids'], batch['attention_mask'])
                labels.append(batch['labels'].cpu().numpy())
                probs.extend(outputs['predictions'].float().cpu().numpy())
                attn_scores.extend(outputs['attention_scores'].float().cpu().numpy())
                sampled_chromosomes.extend(batch['sampled_chromosomes'])

        N -= 1
        if N <= 0:
            break

    return {
        'labels': np.concatenate(labels),
        'probs': np.array(probs),
        'sampled_chromosomes': sampled_chromosomes,
        'attn_scores': np.array(attn_scores)
    }

def run_inference(args):
    # Load model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer()
    gender_model = GenderChunkedClassifier(base_model)
    load_model(gender_model, args.model_path)
    gender_model = gender_model.cuda()
    gender_model.eval()

    # Load dataset
    df = pd.read_csv(f"{args.data_dir}/{args.split}_merged_metadata.csv", index_col=0)

    results = {
        'sample_ids_labels': {},
        'sample_ids_sampled_chromosomes': {},
        'sample_ids_probs': {},
        'sample_ids_attn_scores': {}
    }

    if args.force_species: 
        species_list = [args.force_species]
    else:
        species_list = list(df['organism_name'])
    for species in species_list:
        logger.info(f"Inferring for species {species}...")

        dataset = MultiSpeciesGenderDataChunkedDataset(
            data_path=args.data_dir,
            split_name=args.split,
            n_chunks=args.n_chunks,
            chunk_size=args.chunk_size,
            force_sampling_from_y=args.force_sampling_from_y,
            force_species=species,
            chrY_ratio=args.chrY_ratio,
            chrX_ratio=args.chrX_ratio,
            seed=args.seed+1
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=lambda x: preprocess_collate_fn(x, tokenizer),
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        N = args.n_per_sample // args.batch_size
        batch_results = process_batches(dataloader, gender_model, N, species)
        
        # Store results
        for key in results:
            results[key][species] = batch_results[key.replace('sample_ids_', '')]

        # Print sample results
        probs = results['sample_ids_probs'][species]
        logger.info(f'{species}: {len(probs)} samples \n'
                   f'threshold: {(probs>0.5).sum() / len(probs):.2f}')

    # Save results
    if not args.force_species:
        species = 'all'

    output_filename = f'{args.output_file_prefix}_model_{args.model_path.replace("/", "_")}_{args.split}_species_{species}_force_Y_sampling_{args.force_sampling_from_y}_Y_ratio_{args.chrY_ratio}_X_ratio_{args.chrX_ratio}_{args.n_per_sample}_per_sample_{args.seed}.pckl'
    if args.save_probs:
        with open(output_filename, 'wb') as f:
            pickle.dump({
                'sample_ids_probs': results['sample_ids_probs'],
                'sample_ids_labels': results['sample_ids_labels'],
                'sample_ids_sampled_chromosomes': results['sample_ids_sampled_chromosomes']
            }, f)
    
    return results['sample_ids_probs'], results['sample_ids_labels'], results['sample_ids_sampled_chromosomes']

def run_evaluation(sample_ids_probs, sample_ids_labels, sample_ids_sampled_chromosomes, args):

    for k in sample_ids_sampled_chromosomes:
        sample_ids_sampled_chromosomes[k] = np.array(sample_ids_sampled_chromosomes[k])
    
    # collating for different genders
    species = list(sample_ids_probs.keys())[0]
    new_sample_ids_probs = {species+"_male": [], species+"_female": [] }
    new_sample_ids_sampled_chromosomes = {species+"_male": [], species+"_female": [] }
    new_sample_ids_labels = {species+"_male": 0, species+"_female": 1}

    for i, elem in enumerate(sample_ids_probs[species]):
        if sample_ids_labels[species][i] == 0:
            new_sample_ids_probs[species+"_male"].append(elem)
            new_sample_ids_sampled_chromosomes[species+"_male"].append(sample_ids_sampled_chromosomes[species][i])

        elif sample_ids_labels[species][i] == 1:
            new_sample_ids_probs[species+"_female"].append(elem)
            new_sample_ids_sampled_chromosomes[species+"_female"].append(sample_ids_sampled_chromosomes[species][i])

    for key in new_sample_ids_probs.keys():
        new_sample_ids_probs[key] = np.array(new_sample_ids_probs[key]).reshape(-1, 1)
    for key in new_sample_ids_sampled_chromosomes.keys():
        new_sample_ids_sampled_chromosomes[key] = np.array(new_sample_ids_sampled_chromosomes[key])

    max_N = args.n_per_sample
    Ns = []
    for N in [25, 100, 1_000, 5_000, 15_000, 30_000, 60_000]:
        if N < max_N:
            Ns.append(N)
    Ns.append(max_N)
    N2thr = find_threshold_for_N(new_sample_ids_labels, new_sample_ids_probs, Ns=Ns)
    
    accs, stds, X_probs, Y_probs = calculate_metrics(new_sample_ids_labels, new_sample_ids_probs, new_sample_ids_sampled_chromosomes, N2threshold=N2thr)
    with open(f"{args.metrics_output_file_prefix}", "w") as f:
        f.write(json.dumps({
            "checkpoint": args.model_path,
            "evaluation_species": species,
            "N2thr": N2thr,
            "Ns": Ns,
            "accs": accs,
            "stds": stds,
            "n_chunks": args.n_chunks,
            "chunk_size": args.chunk_size,
            "n_per_sample": args.n_per_sample,
            "force_sampling_from_y": args.force_sampling_from_y,
            "chrY_ratio": args.chrY_ratio,
            "chrX_ratio": args.chrX_ratio,
            "seed": args.seed,
        }) + "\n")

    return Ns, accs, stds

if __name__ == '__main__':
    args = parse_args()
    sample_ids_probs, sample_ids_labels, sample_ids_sampled_chromosomes = run_inference(args)
    Ns, accs, stds = run_evaluation(sample_ids_probs, sample_ids_labels, sample_ids_sampled_chromosomes, args)
    logger.info(f'At N = {Ns} samples:')
    logger.info(f'Accuracy: {accs}')
    logger.info(f'Standard deviation: {stds}')
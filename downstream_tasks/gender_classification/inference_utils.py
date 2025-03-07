import os
import pickle
import numpy as np
from tqdm.auto import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from gender_dataset import MultiSpeciesGenderDataChunkedDataset, worker_init_fn, collate_fn


def infer(gender_model, tokenizer, split_name, n_chunks=16, chunk_size=3072, seed=142, batch_size=64, 
            chrX_ratio = None, chrY_ratio = None, force_sampling_from_y=False):

    gender_model = gender_model.cuda()

    dataset = MultiSpeciesGenderDataChunkedDataset(split_name=split_name, n_chunks=n_chunks,
                                        chunk_size=chunk_size,
                                        force_sampling_from_y=force_sampling_from_y,
                                        chrY_ratio=chrY_ratio,
                                        chrX_ratio=chrX_ratio,
                                        seed=seed+1)


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

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=preprocess_collate_fn,
                          num_workers=1, pin_memory=True,
                          worker_init_fn=worker_init_fn)
    
    N = 100000 // batch_size

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
    
    return sample_ids_labels, sample_ids_probs, sample_ids_sampled_chromosomes


def load_generated_output(dumps):
    sample_ids_probs = None
    sample_ids_labels = None
    sample_ids_sampled_chromosomes = None

    for dump in dumps:

        d = pickle.load(open(dump, 'rb'))

        if sample_ids_labels is None:
            sample_ids_labels = d['sample_ids_labels']

        if sample_ids_probs is None:
            sample_ids_probs = d['sample_ids_probs']

        if sample_ids_sampled_chromosomes is None:
            sample_ids_sampled_chromosomes = d['sample_ids_sampled_chromosomes']
            
        else:
            sample_ids_labels.update(d['sample_ids_labels'])

            for k in d['sample_ids_probs']:
                if k in sample_ids_probs:
                    sample_ids_probs[k] = np.concatenate([sample_ids_probs[k], d['sample_ids_probs'][k]])
                else:
                    sample_ids_probs[k] = d['sample_ids_probs'][k]
            
            for k in d['sample_ids_sampled_chromosomes']:
                if k in sample_ids_sampled_chromosomes:
                    sample_ids_sampled_chromosomes[k] = np.concatenate([sample_ids_sampled_chromosomes[k], d['sample_ids_sampled_chromosomes'][k]])
                else:
                    sample_ids_sampled_chromosomes[k] = d['sample_ids_sampled_chromosomes'][k]

    return sample_ids_probs, sample_ids_labels, sample_ids_sampled_chromosomes


def find_threshold_for_N(sample_ids_labels, sample_ids_probs, Ns=[25, 50, 100, 200, 500, 1000, 5000, 8000, 15000, 30000], sampling_freq=200):

    sample_ids_sorted = [el[0] for el in sorted(sample_ids_labels.items(), key=lambda x: x[1])]
    
    THR = 0.5

    N2thr = {}

    for N in Ns:
        ratios = []
        for i in range(sampling_freq):
            
            for k in sample_ids_sorted:
                probs = sample_ids_probs[k]
                
                # sampling N probabilities from N chunks
                prob_idxs = np.random.choice(list(range(len(probs[:,0]))), size=min(N, len(probs)))
                probs = probs[:,0][prob_idxs]
                ratios += [(probs > THR).sum() / len(probs)]                

        ratios = np.array(ratios)
        N2thr[N] = np.median(ratios)
    
    return N2thr


def calculate_metrics(sample_ids_labels, sample_ids_probs, sample_ids_sampled_chromosomes, N2threshold, sampling_freq=200):
    
    Ns = list(N2threshold.keys())
    sample_ids_sorted = [el[0] for el in sorted(sample_ids_labels.items(), key=lambda x: x[1])]
    THR = 0.5

    accs = []
    stds = []
    X_probs = []
    Y_probs = []

    for N in Ns:
        N_accs = []
        Ns_containX = []
        Ns_containY = []

        for i in range(sampling_freq):
            ratios = []
            for k in sample_ids_sorted:
                probs = sample_ids_probs[k]
                
                # sampling N probabilities from N chunks
                prob_idxs = np.random.choice(list(range(len(probs[:,0]))), size=min(N, len(probs)))
                probs = probs[:,0][prob_idxs]
                ratios += [(probs > THR).sum() / len(probs)]

                chrs = [sample_ids_sampled_chromosomes[k][idx] for idx in prob_idxs]
                Ns_containX.append(any(['chrX' in elem for elem in chrs]))
                if sample_ids_labels[k] == 0:
                    Ns_containY.append(any(['chrY' in elem for elem in chrs]))
                
                
            ratios = np.array(ratios)
            
            acc = 0
            for k, r in zip(sample_ids_sorted, ratios):
                acc += int(int(r > N2threshold[N]) == sample_ids_labels[k])
            
            acc = acc / len(sample_ids_sorted)
            N_accs += [acc]

        accs += [np.mean(N_accs)]
        stds.append(np.std(N_accs))
        X_probs.append(np.mean(Ns_containX))
        Y_probs.append(np.mean(Ns_containY))

    return accs, stds, X_probs, Y_probs
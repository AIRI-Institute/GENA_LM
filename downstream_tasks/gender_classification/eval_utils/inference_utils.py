import os
import pickle
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import torch
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast

# from gender_dataset import MultiSpeciesGenderDataChunkedDataset, worker_init_fn, collate_fn


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


def find_threshold_for_N(sample_ids_labels, sample_ids_probs, Ns=[25, 50, 100, 200, 500, 1_000, 5_000, 8_000, 15_000, 30_000], sampling_freq=200):

    sample_ids_sorted = [el[0] for el in sorted(sample_ids_labels.items(), key=lambda x: x[1])]
    
    THR = 0.5

    N2thr = {}

    for N in Ns:
        ratios = []
        for _ in range(sampling_freq):
            
            for k in sample_ids_sorted:
                probs = sample_ids_probs[k]
                
                # sampling N probabilities from N chunks
                prob_idxs = np.random.choice(list(range(len(probs[:,0]))), size=min(N, len(probs)))
                probs = probs[:,0][prob_idxs]
                ratios += [(probs > THR).sum() / len(probs)]                

        ratios = np.array(ratios)
        N2thr[N] = np.quantile(ratios, 1 - np.mean(list(sample_ids_labels.values())))
    
    return N2thr


def calculate_metrics(sample_ids_labels, sample_ids_probs, sample_ids_sampled_chromosomes, N2threshold, sampling_freq=200, metric_func=accuracy_score):
    
    Ns = list(N2threshold.keys())
    sample_ids_sorted = [el[0] for el in sorted(sample_ids_labels.items(), key=lambda x: x[1])]
    THR = 0.5

    scores = []
    stds = []
    X_probs = []
    Y_probs = []

    for N in Ns:
        N_scores = []
        Ns_containX = []
        Ns_containY = []

        for _ in tqdm(range(sampling_freq), total=len(range(sampling_freq))):
            ratios = []
            for k in sample_ids_sorted:
                probs = sample_ids_probs[k]
                
                # sampling N probabilities from N chunks
                prob_idxs = np.random.choice(list(range(len(probs[:,0]))), size=min(N, len(probs)))
                probs = probs[:,0][prob_idxs]
                ratios += [(probs > THR).sum() / len(probs)]

                chrs = sample_ids_sampled_chromosomes[k][prob_idxs, :, 0]
                
                Ns_containX.append(any([any(['X' in chr for chr in chunk ]) for chunk in chrs]))
                if sample_ids_labels[k] == 0:
                    Ns_containY.append(any([any(['Y' in chr for chr in chunk ]) for chunk in chrs]))
                
            ratios = np.array(ratios)
            
            preds = []
            labels = []
            for k, r in zip(sample_ids_sorted, ratios):
                preds.append(int(r > N2threshold[N]))
                labels.append(sample_ids_labels[k])
                # acc += int(int(r > N2threshold[N]) == sample_ids_labels[k])
                
            
            score = metric_func(y_pred=preds, y_true=labels)
            N_scores += [score]
            # N_accs += [acc]

        scores += [np.mean(N_scores)]
        stds.append(np.std(N_scores))
        X_probs.append(np.mean(Ns_containX))
        Y_probs.append(np.mean(Ns_containY))

    return scores, stds, X_probs, Y_probs
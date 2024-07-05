import torch
import torch.distributed
from torch.utils.data import IterableDataset
import h5py
import pandas as pd
import numpy as np
from pathlib import Path


class GenderDataChunkedDataset(IterableDataset):
    """
    usage example:

    data_path = '~/gender_data'
    split_name = 'train'
    split_data_path = Path(data_path) / f'{split_name}.h5'
    split_labels_path = Path(data_path) / f'{split_name}.csv'
    seed = 142

    dataset = RandomSampleIterableDataset(split_data_path, split_labels_path, n_chunks=5, chunk_size=10, seed=seed)

    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    """
    def __init__(self, data_path, labels_path, n_chunks=128, chunk_size=512, max_n_samples=None, seed=None):
        # chunk_size is in base pairs
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path)
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.max_n_samples = max_n_samples
        self.seed = seed

        self.epoch = 0

        # 1 - male, 2 - female
        self.labels = pd.read_csv(self.labels_path, index_col=0).set_index('sample')

        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def get_chunk_from_sample(self, sample_id):
        sample_data = self.data[sample_id]
        # get chromosome lengths for the selected sample
        chr_lengths = {k: sample_data[k].shape[0] for k in sample_data.keys()}
        idx_to_chr = [k for k in chr_lengths]
        total_bp = sum(chr_lengths.values())

        # get chromosome sampling probabilities
        probs = [chr_lengths[k] / total_bp for k in chr_lengths]

        # about 1% chance to sample from whole genome one chunk from chrY
        sampled_chrs = [np.random.choice(idx_to_chr, p=probs) for _ in range(self.n_chunks)]
        chunks = []
        for chr in sampled_chrs:
            start = np.random.randint(0, chr_lengths[chr] - self.chunk_size)
            chunk = ''.join(sample_data[chr][start:start + self.chunk_size].astype(str))
            chunks.append(chunk)

        return chunks, sampled_chrs, chr_lengths

    def __iter__(self):
        # read the data once per worker (not to share h5py file object between workers)
        self.data = h5py.File(self.data_path, 'r')
        self.sample_ids = list(self.data.keys())
        n_iters = 0
        while True:
            if self.max_n_samples is not None:
                if n_iters > self.max_n_samples:
                    break
            n_iters += 1

            # randomly select a sample_id
            sample_id = np.random.choice(self.sample_ids)

            chunks, sampled_chrs, chr_lengths = self.get_chunk_from_sample(sample_id)

            yield {
                'labels': self.labels.loc[sample_id]['sex'] - 1,
                'sample_id': sample_id,
                'sampled_chromosomes': sampled_chrs,
                'chunks': chunks,
                'full_sample_has_y_chr': 'chrY' in chr_lengths,
                'has_y_chr_sampled': 'chrY' in sampled_chrs,
                }


def collate_fn(samples):
    keys = [k for k in samples[0]]
    batch = {}
    for k in keys:
        batch[k] = [sample[k] for sample in samples]
    return batch


def worker_init_fn(worker_id):
    # set different seeds for all workers to generate different samples
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # initialize the seed for each worker
    seed = dataset.seed + worker_id + global_rank * worker_info.num_workers if dataset.seed is not None else None
    dataset.set_seed(seed)

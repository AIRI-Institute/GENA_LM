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
    def __init__(self, data_path, labels_path, n_chunks=128, chunk_size=512, force_sampling_from_y=False,
                 chrY_name='chrY', label_column='sex', sample_column='sample',
                 chrY_ratio=None, max_n_samples=None, seed=None):
        # chunk_size is in base pairs
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path)
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.force_sampling_from_y = force_sampling_from_y
        self.chrY_name = chrY_name
        self.label_column = label_column
        self.sample_column = sample_column
        self.chrY_ratio = chrY_ratio
        self.max_n_samples = max_n_samples
        self.seed = seed

        # 1 - male, 2 - female  # in human data
        # M - male, F - female  # in mouse data
        # map labels to
        # 0 - male, 1 - female
        labels_map = {1: 0, 2: 1, 'M': 0, 'F': 1}

        self.labels = pd.read_csv(self.labels_path, index_col=0).set_index(self.sample_column)
        self.labels[self.label_column] = self.labels[self.label_column].map(labels_map)

        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def filter_chr_names(self, names, chrY_name):
        # choose only one from [chrY, chrY_with_SNPs] as chrY
        chrY_names = [k for k in names if 'chrY' in k]
        non_chrY_names = [k for k in names if 'chrY' not in k]

        if chrY_name in chrY_names:
            chrY_names = [chrY_name]
        else:
            chrY_names = []
        return chrY_names + non_chrY_names

    def get_chunks_from_sample(self, sample_id):
        sample_data = self.data[sample_id]
        chr_names = self.filter_chr_names(sample_data.keys(), self.chrY_name)
        # get chromosome lengths for the selected sample
        chr_lengths = {k: sample_data[k].shape[0] for k in chr_names}
        total_bp = sum(chr_lengths.values())

        # get chromosome sampling probabilities
        chr_probs = {k: chr_lengths[k] / total_bp for k in chr_lengths}

        # upsample/subsample chrY if self.chrY_ratio is set
        if self.chrY_name in chr_lengths and self.chrY_ratio is not None:
            old_val = chr_probs[self.chrY_name]
            chr_probs[self.chrY_name] = self.chrY_ratio
            for k in chr_probs:
                if k != self.chrY_name:
                    chr_probs[k] *= (1 - self.chrY_ratio) / (1 - old_val)

        idx_to_chr, probs = zip(*chr_probs.items())

        # about 1% chance to sample from whole genome one chunk from chrY
        if self.force_sampling_from_y and self.chrY_name in chr_lengths:
            while True:
                sampled_chrs = [np.random.choice(idx_to_chr, p=probs) for _ in range(self.n_chunks)]
                if self.chrY_name in sampled_chrs:
                    break
        else:
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

            chunks, sampled_chrs, chr_lengths = self.get_chunks_from_sample(sample_id)

            yield {
                'labels': self.labels.loc[sample_id][self.label_column],
                'sample_id': sample_id,
                'sampled_chromosomes': sampled_chrs,
                'chunks': chunks,
                'full_sample_has_y_chr': self.chrY_name in chr_lengths,
                'has_y_chr_sampled': self.chrY_name in sampled_chrs,
                }


class SpeciesSampler:
        def __init__(self, data_path, split_name="train", force_sampling_from_y=False,
                 chrY_name='chrY', label_column='sex', sample_column='sample',
                 chrY_ratio=0.1, seed=None):
                
                # chunk_size is in base pairs
                self.data_path = Path(data_path + "/" + split_name + ".h5")
                self.labels_path = Path(data_path + "/" + split_name + ".csv")
                self.size = len(h5py.File(self.data_path, 'r'))

                self.label_column = label_column
                self.sample_column = sample_column
                
                self.chrY_name = chrY_name
                self.chrY_ratio = chrY_ratio
                self.force_sampling_from_y = force_sampling_from_y
                
                self.seed = seed

                # 1 - male, 2 - female  # in human data
                # M - male, F - female  # in mouse data
                # map labels to
                # 0 - male, 1 - female
                labels_map = {1: 0, 2: 1, 'M': 0, 'F': 1}

                self.labels = pd.read_csv(self.labels_path, index_col=0).set_index(self.sample_column)
                self.labels[self.label_column] = self.labels[self.label_column].map(labels_map)

                self.set_seed(seed)

        def set_seed(self, seed):
                self.seed = seed
                np.random.seed(seed)

        def filter_chr_names(self, names, chrY_name):
                # choose only one from [chrY, chrY_with_SNPs] as chrY
                chrY_names = [k for k in names if 'chrY' in k]
                non_chrY_names = [k for k in names if 'chrY' not in k]

                if chrY_name in chrY_names:
                        chrY_names = [chrY_name]
                else:
                        chrY_names = []
                
                return chrY_names + non_chrY_names


        def get_species_chunk(self, chunk_size, n_chunks):
                
                # choosing random sample
                self.data = h5py.File(self.data_path, 'r')
                self.sample_ids = list(self.data.keys())
                sample_id = np.random.choice(self.sample_ids)

                sample_data = self.data[sample_id]
                chr_names = self.filter_chr_names(sample_data.keys(), self.chrY_name)

                # get chromosome lengths for the selected sample
                chr_lengths = {k: sample_data[k].shape[0] for k in chr_names}
                total_bp = sum(chr_lengths.values())

                # get chromosome sampling probabilities
                chr_probs = {k: chr_lengths[k] / total_bp for k in chr_lengths}

                # upsample/subsample chrY if self.chrY_ratio is set
                if self.chrY_name in chr_lengths and self.chrY_ratio is not None:
                        old_val = chr_probs[self.chrY_name]
                        chr_probs[self.chrY_name] = self.chrY_ratio
                        
                        for k in chr_probs:
                                if k != self.chrY_name:
                                        chr_probs[k] *= (1 - self.chrY_ratio) / (1 - old_val)

                idx_to_chr, probs = zip(*chr_probs.items())

                # about 1% chance to sample from whole genome one chunk from chrY
                if self.force_sampling_from_y and self.chrY_name in chr_lengths:
                        while True:
                                sampled_chrs = [np.random.choice(idx_to_chr, p=probs) for _ in range(n_chunks)]
                                if self.chrY_name in sampled_chrs:
                                        break
                else:
                        sampled_chrs = [np.random.choice(idx_to_chr, p=probs) for _ in range(n_chunks)]

                chunks = []
                for chr in sampled_chrs:
                        start = np.random.randint(0, chr_lengths[chr] - chunk_size)
                        chunk = ''.join(sample_data[chr][start:start + chunk_size].astype(str))
                        chunks.append(chunk)
                
                labels = self.labels.loc[sample_id][self.label_column]
                full_sample_has_y_chr = self.chrY_name in chr_lengths
                has_y_chr_sampled = self.chrY_name in sampled_chrs

                return labels, sample_id, sampled_chrs, chunks, full_sample_has_y_chr, has_y_chr_sampled


class MultiSpeciesGenderDataChunkedDataset(IterableDataset):
    def __init__(self, split_name='train', n_chunks=128, chunk_size=512, force_sampling_from_y=False, 
                 chrY_ratio=None, max_n_samples=None, seed=None):
        
        self.max_n_samples = max_n_samples
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        
        self.common_params = {
            "split_name": split_name,
            "force_sampling_from_y": force_sampling_from_y,
            "chrY_ratio": chrY_ratio, 
            "seed": seed
        }
        
        self.species2metadata = {"homo_sapiens": 
                                    {
                                        "data_path": "/mnt/20tb/ykuratov/gender_data/",
                                        "chrY_name": "chrY",
                                        "sample_column": "sample",
                                        "label_column": "sex"
                                    },
                                "mus_musculus":
                                    {
                                        "data_path": "/mnt/20tb/ykuratov/mouse_gender_data/",
                                        "chrY_name": "chrY",
                                        "sample_column": "strain_name",
                                        "label_column": "gender"
                                    }
                                }

        self.species2sampler = {}
        species2size = {}
        
        for species in self.species2metadata:
            self.species2sampler[species] = SpeciesSampler(**self.species2metadata[species], **self.common_params)

            species2size[species] = self.species2sampler[species].size
                        
        total_samples = sum(species2size.values())
        self.species2prob = {species: species2size[species] / total_samples for species in species2size}
                
        self.set_seed(seed)


    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def __iter__(self):
        # read the data once per worker (not to share h5py file object between workers)

        species, probs = zip(*self.species2prob.items())

        n_iters = 0
        while True:
            if self.max_n_samples is not None:
                if n_iters > self.max_n_samples:
                    break
            n_iters += 1
            
            # randomly select a species and its sample_id

            sampled_species = np.random.choice(species, p=probs)

            labels, sample_id, sampled_chrs, chunks, \
                full_sample_has_y_chr, has_y_chr_sampled = self.species2sampler[sampled_species].get_species_chunk(chunk_size=self.chunk_size, n_chunks=self.n_chunks)


            yield {
                'labels': labels,
                'species': sampled_species,
                'sample_id': sample_id,
                'sampled_chromosomes': sampled_chrs,
                'chunks': chunks,
                'full_sample_has_y_chr': full_sample_has_y_chr,
                'has_y_chr_sampled': has_y_chr_sampled,
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

    # if max_n_samples is set update it acording to number of dataloader workers
    if dataset.max_n_samples is not None:
        dataset.max_n_samples = dataset.max_n_samples // worker_info.num_workers

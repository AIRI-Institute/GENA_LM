import torch
import torch.distributed
from torch.utils.data import IterableDataset
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import os

log_lvl = logging.INFO
logging.basicConfig(level=log_lvl)
logger = logging.getLogger('mammals_gender_dataset')


class MultiSpeciesGenderDataChunkedDataset(IterableDataset):
    def __init__(self, data_path, split_name='train', n_chunks=128, chunk_size=512, chrY_name='Y', chrX_name='X',
                 chrY_ratio=None, chrX_ratio=None, force_sampling_from_y=False, max_n_samples=None, seed=None, force_species=None):
        
        # self.data_path = Path(os.path.join(data_path, 'merged_links_' + split_name + '.h5'))
        self.data_path = Path(os.path.join(data_path, split_name + '.h5'))
        self.metadata_df = pd.read_csv(Path(os.path.join(data_path, split_name + '_merged_metadata.csv')))
        
        with open(Path(os.path.join(data_path, 'split2organism_name.json')), 'r') as f:
            split2organism_name = json.load(f)
        self.split2organism_name = split2organism_name


        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.max_n_samples = max_n_samples
        
        self.chrY_name = chrY_name
        self.chrY_ratio = chrY_ratio
        self.force_sampling_from_y = force_sampling_from_y

        self.chrX_name = chrX_name
        self.chrX_ratio = chrX_ratio
        self.force_species = force_species

        self.species = self.split2organism_name[split_name]
        self.set_seed(seed)

        if self.force_species is not None and self.force_species not in self.species:
            raise ValueError(f"Species '{self.force_species}' not found in {split_name} split. "
                           f"Available species: {self.species}")

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)


    def __iter__(self):
        # read the data once per worker (not to share h5py file object between workers)
        self.data = h5py.File(self.data_path, 'r')
        
        n_iters = 0
        while True:
            if self.max_n_samples is not None:
                if n_iters > self.max_n_samples:
                    break
            n_iters += 1
            
            # randomly select a species
            if not self.force_species:
                sampled_species = np.random.choice(self.species)
            else:
                sampled_species = self.force_species
            
            sampled_id = np.random.choice(self.metadata_df[self.metadata_df['organism_name'] == sampled_species]['assembly_accession/sample_id'].values)
            
            label = self.metadata_df[self.metadata_df['assembly_accession/sample_id'] == sampled_id]['sex'].values[0]
            
            # if label is not set, randomly select a sex
            if np.isnan(label):
                label = np.random.randint(0, 2)

            # we need to know if the sample is diploid or not to calculate chromosome ratio properly
            diploid = self.metadata_df[self.metadata_df['assembly_accession/sample_id'] == sampled_id]['diploid'].values[0]

            logger.debug(f"Sample organism name: {sampled_species}")
            logger.debug(f"Sample/assembly id: {sampled_id}")
            logger.debug(f"Diploid: {diploid}")
            logger.debug(f"Label: {label}")

            sampled_chrs, chunks, full_sample_has_y_chr, has_y_chr_sampled, has_x_chr_sampled = self.get_chunks_from_sample(sampled_id, label, diploid)

            yield {
                'sample_ids': sampled_id,
                'labels': label,
                'species': sampled_species,
                'sampled_chromosomes': sampled_chrs,
                'chunks': chunks,
                'full_sample_has_y_chr': full_sample_has_y_chr,
                'has_y_chr_sampled': has_y_chr_sampled,
                'has_x_chr_sampled': has_x_chr_sampled
                }

    def filter_chr_names(self, names, label):

        chrY_names = [k for k in names if self.chrY_name in k]

        if 'chrY_with_SNPs' in chrY_names:
             chrY_names = ["chrY_with_SNPs"]

        non_chrY_names = [k for k in names if self.chrY_name not in k]

        if label == 0:
            return chrY_names + non_chrY_names
        else:
            return non_chrY_names


    def get_chunks_from_sample(self, sampled_id, label, diploid=False):

        sample_data = self.data[sampled_id]
        
        # filter chromosomes by sex
        chr_names = self.filter_chr_names(sample_data.keys(), label)

        sex_chromosomes = [chr for chr in chr_names if (self.chrY_name in chr) or (self.chrX_name in chr)]
        logger.debug(f'Sex chromosomes: {sex_chromosomes}')

        autosomes = [chr for chr in chr_names if chr not in sex_chromosomes]
        # print(autosomes)
        logger.debug(f'Autosomes: {autosomes}')

        if not diploid:
            autosome_lengths = {k: sample_data[k].shape[0] * 2 for k in autosomes}
            # for males we have both X and Y chromosomes, so we can preserve X and Y chromosomes' lengths
            if label == 0:
                sex_chromosome_lengths = {k: sample_data[k].shape[0] for k in sex_chromosomes}                
            else:
                # for females we have only X chromosome, so we double it length
                sex_chromosome_lengths = {k: sample_data[k].shape[0] * 2 for k in sex_chromosomes}
        else:
            autosome_lengths = {k: sample_data[k].shape[0] for k in autosomes}
            sex_chromosome_lengths = {k: sample_data[k].shape[0] for k in sex_chromosomes}

        autosome_bp = sum(autosome_lengths.values())
        sex_chromosome_bp = sum(sex_chromosome_lengths.values())

        total_bp = autosome_bp + sex_chromosome_bp
        chr_lengths = {**sex_chromosome_lengths, **autosome_lengths}
            
        logger.debug(f"Chromosomes length: {chr_lengths}")
 
        # get initial chromosome sampling probabilities
        chr_probs = {k: chr_lengths[k] / total_bp for k in chr_lengths}
        
        # Warning! For chrY_ratio and chrX_ratio set simultaneously it won't work properly
        # Ratio parameters are for debugging purposes only
        # upsample/subsample chrY if self.chrY_ratio is set
        if label == 0 and self.chrY_ratio is not None:

            chrY_names = [chr for chr in sex_chromosomes if self.chrY_name in chr]

            chrY_len_sum = sum([chr_lengths[k] for k in chrY_names])
            old_Y_prob = sum([chr_probs[k] for k in chrY_names])

            for k in chrY_names:
                chr_probs[k] = self.chrY_ratio * chr_lengths[k] / chrY_len_sum
            
            for k in chr_probs:
                if k not in chrY_names:
                    chr_probs[k] *= (1 - self.chrY_ratio) / (1 - old_Y_prob)

        # upsample/subsample chrX if self.chrX_ratio is set
        if self.chrX_ratio is not None:
            chrX_names = [chr for chr in sex_chromosomes if self.chrX_name in chr]

            chrX_len_sum = sum([chr_lengths[k] for k in chrX_names])
            old_X_prob = sum([chr_probs[k] for k in chrX_names])

            for k in chrX_names:
                chr_probs[k] = self.chrX_ratio * chr_lengths[k] / chrX_len_sum
                
            for k in chr_probs:
                if k not in chrX_names:
                    chr_probs[k] *= (1 - self.chrX_ratio) / (1 - old_X_prob)

        
        logger.debug(f"Chromosomes probabilities: {chr_probs}")
        logger.debug(f"Chromosomes probabilities sum: {sum(chr_probs.values())}")

        idx_to_chr, probs = zip(*chr_probs.items())
    
        # about 1% chance to sample from whole genome one chunk from chrY
        if self.force_sampling_from_y and any([self.chrY_name in chr for chr in sex_chromosomes]):
            while True:
                sampled_chrs = [np.random.choice(idx_to_chr, p=probs) for _ in range(self.n_chunks)]
                if any([self.chrY_name in chr for chr in sampled_chrs]):
                    break
        else:
            sampled_chrs = [np.random.choice(idx_to_chr, p=probs) for _ in range(self.n_chunks)]

        chunks = []
        for i, chr in enumerate(sampled_chrs):
            if sample_data[chr].shape[0] > self.chunk_size:
                start = np.random.randint(0, sample_data[chr].shape[0] - self.chunk_size)
                chunk = sample_data[chr][start:start + self.chunk_size].tobytes().decode('ascii')
                sampled_chrs[i] = (chr, start, start + self.chunk_size)
            else:
                chunk = sample_data[chr][:].tobytes().decode('ascii')
                sampled_chrs[i] = (chr, 0, sample_data[chr].shape[0])
            # if "Y" in chr:
            #     print(chunk)
            chunks.append(chunk)
            # assert len(chunk) == chunk_size

        full_sample_has_y_chr = any([self.chrY_name in chr for chr in chr_lengths])
        has_y_chr_sampled = any([self.chrY_name in chr for chr in sampled_chrs])
        has_x_chr_sampled = any([self.chrX_name in chr for chr in sampled_chrs])

        logger.debug(f"Label: {label}")
        logger.debug(f"Sample has Y chromosome: {full_sample_has_y_chr}")
        logger.debug(f"Y chromosome in sampled chromosomes: {has_y_chr_sampled}")
        logger.debug(f"X chromosome in sampled chromosomes: {has_x_chr_sampled}")

        return sampled_chrs, chunks, full_sample_has_y_chr, has_y_chr_sampled, has_x_chr_sampled


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

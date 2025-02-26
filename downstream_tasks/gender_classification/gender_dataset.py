import torch
import torch.distributed
from torch.utils.data import IterableDataset
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import logging

log_lvl = logging.INFO
logging.basicConfig(level=log_lvl)
logger = logging.getLogger('')

class SpeciesSampler:
    def __init__(self, data_path, split_name="train", chromosome_number=42, force_sampling_from_y=False,
            chrX_name='chrX', chrY_name='chrY', label_column='sex', sample_column='sample',
            chrX_ratio=None, chrY_ratio=None, seed=None):

            self.chromosome_number = chromosome_number
            self.data_path = Path(data_path + "/" + split_name + ".h5")
            self.labels_path = Path(data_path + "/" + split_name + ".csv")

            self.label_column = label_column
            self.sample_column = sample_column
            
            self.chrY_name = chrY_name
            self.chrY_ratio = chrY_ratio
            self.force_sampling_from_y = force_sampling_from_y

            self.chrX_name = chrX_name
            self.chrX_ratio = chrX_ratio
            
            self.seed = seed

            # 1 - male, 2 - female  # in human data
            # M - male, F - female  # in mouse data
            # map labels to
            # 0 - male, 1 - female
            labels_map = {1: 0, 2: 1, 'M': 0, 'F': 1}

            self.labels = pd.read_csv(self.labels_path, index_col=0).set_index(self.sample_column)
            self.labels[self.label_column] = self.labels[self.label_column].map(labels_map)

            self.data = h5py.File(self.data_path, 'r')
            self.size = len(self.data)

            self.set_seed(seed)

    def set_seed(self, seed):
            self.seed = seed
            np.random.seed(seed)

    def filter_chr_names(self, names):
            # choose only one from [chrY, chrY_with_SNPs] as chrY
            chrY_names = [k for k in names if 'chrY' in k]
            non_chrY_names = [k for k in names if 'chrY' not in k]

            if self.chrY_name in chrY_names:
                chrY_names = [self.chrY_name]

            elif len(chrY_names) > 0:
                chrY_names = [chrY_names[0]]
                
            else:
                chrY_names = []

            return chrY_names + non_chrY_names
    
    def get_species_chunk(self, chunk_size, n_chunks):
            
            # choosing random sample
            self.sample_ids = list(self.data.keys())
            sample_id = np.random.choice(self.sample_ids)

            logger.debug(f"Sample id: {sample_id}")

            sample_data = self.data[sample_id]

            # in case there is chrY with SNP, we need to stick with only one chrY
            chr_names = self.filter_chr_names(sample_data.keys())

            sex_chromosomes = [chr for chr in chr_names if (self.chrY_name in chr) or (self.chrX_name in chr)]
            logger.debug(f'Sex chromosomes: {sex_chromosomes}')

            autosomes = [chr for chr in chr_names if chr not in sex_chromosomes]
            logger.debug(f'Autosomes: {autosomes}')

            # get chromosome lengths for the selected sample

            # if diploid chromosome set
            if len(autosomes) + len(sex_chromosomes) == self.chromosome_number:
                logger.debug(f"Diploid chromosomes set")
                chr_lengths = {k: sample_data[k].shape[0] for k in chr_names}
                total_bp = sum(chr_lengths.values())

            # if haploid chromosome set
            else:
                logger.debug(f"Haploid chromosomes set")
                autosome_lengths = {k: sample_data[k].shape[0] * 2 for k in autosomes}
                autosome_bp = sum(autosome_lengths.values())

                # for male we can just sum up the length of sex chromosomes as both X and Y are already presented
                if any([self.chrY_name in chr for chr in sex_chromosomes]):
                    sex_chromosome_lengths = {k: sample_data[k].shape[0] for k in sex_chromosomes}
                
                # for females in case of haploid chromosome set we need to double the length of X chromosome
                else:
                    # if only one X chromosomes are presented in case of a female sample - we double it
                    sex_chromosome_lengths = {k: sample_data[k].shape[0] * 2 for k in sex_chromosomes}
                    
                sex_chromosome_bp = sum(sex_chromosome_lengths.values())
                total_bp = autosome_bp + sex_chromosome_bp
                chr_lengths = {**sex_chromosome_lengths, **autosome_lengths}
            
            logger.debug(f"Chromosomes length: {chr_lengths}")
            
            # get initial chromosome sampling probabilities
            chr_probs = {k: chr_lengths[k] / total_bp for k in chr_lengths}

            # Warning! For chrY_ratio and chrX_ratio set simultaneously it won't work properly
            # Ratio parameters are for debugging purposes only

            # upsample/subsample chrY if self.chrY_ratio is set
            if any([self.chrY_name in chr for chr in sex_chromosomes]) and self.chrY_ratio is not None:

                chrY_name = [chr for chr in sex_chromosomes if self.chrY_name in chr][0]
                old_val = chr_probs[chrY_name]
                chr_probs[chrY_name] = self.chrY_ratio
                
                for k in chr_probs:
                    if self.chrY_name not in k:
                        chr_probs[k] *= (1 - self.chrY_ratio) / (1 - old_val)

            # upsample/subsample chrX if self.chrX_ratio is set
            if self.chrX_ratio is not None:
                chrXs = [chr for chr in sex_chromosomes if self.chrX_name in chr]
                old_vals = sum([chr_probs[sex_chr] for sex_chr in chrXs])
                
                for chr in chrXs:
                    chr_probs[chr] = self.chrX_ratio / len(chrXs)
                    
                for k in chr_probs:
                    if self.chrX_name not in k:
                        chr_probs[k] *= (1 - self.chrX_ratio) / (1 - old_vals)

            
            logger.debug(f"Chromosomes probabilities: {chr_probs}")
            logger.debug(f"Chromosomes probabilities sum: {sum(chr_probs.values())}")

            idx_to_chr, probs = zip(*chr_probs.items())

            # about 1% chance to sample from whole genome one chunk from chrY
            if self.force_sampling_from_y and any([self.chrY_name in chr for chr in sex_chromosomes]):
                while True:
                    sampled_chrs = [np.random.choice(idx_to_chr, p=probs) for _ in range(n_chunks)]
                    if any([self.chrY_name in chr for chr in sampled_chrs]):
                        break
            else:
                sampled_chrs = [np.random.choice(idx_to_chr, p=probs) for _ in range(n_chunks)]

            chunks = []
            for chr in sampled_chrs:
                    start = np.random.randint(0, chr_lengths[chr] - chunk_size)
                    chunk = ''.join(sample_data[chr][start:start + chunk_size].astype(str))
                    chunks.append(chunk)

            labels = self.labels.loc[sample_id][self.label_column]

            full_sample_has_y_chr = any([self.chrY_name in chr for chr in chr_lengths])

            has_y_chr_sampled = any([self.chrY_name in chr for chr in sampled_chrs])

            has_x_chr_sampled = any([self.chrX_name in chr for chr in sampled_chrs])

            logger.debug(f"Label: {labels}")
            logger.debug(f"Sample has Y chromosome: {full_sample_has_y_chr}")
            logger.debug(f"Y chromosome in sampled chromosomes: {has_y_chr_sampled}")
            logger.debug(f"X chromosome in sampled chromosomes: {has_x_chr_sampled}")

            return labels, sample_id, sampled_chrs, chunks, full_sample_has_y_chr, has_y_chr_sampled, has_x_chr_sampled


class MultiSpeciesGenderDataChunkedDataset(IterableDataset):
    def __init__(self, split_name='train', n_chunks=128, chunk_size=512, force_sampling_from_y=False, 
                chrY_ratio=None, chrX_ratio=None, max_n_samples=None, seed=None):
        
        self.max_n_samples = max_n_samples
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        
        self.common_params = {
            "split_name": split_name,
            "force_sampling_from_y": force_sampling_from_y,
            "chrY_ratio": chrY_ratio, 
            "chrX_ratio": chrX_ratio,
            "seed": seed
        }

        self.species2metadata = {"homo_sapiens": 
                                    {
                                        "data_path": "/home/jovyan/data/downstream_tasks/gender_classification/human_gender_data",
                                        "chromosome_number": 46,
                                        "chrY_name": "chrY_with_SNPs",
                                        "chrX_name": 'chrX',
                                        "sample_column": "sample",
                                        "label_column": "sex"
                                    },
                                "mus_musculus":
                                    {
                                        "data_path": "/home/jovyan/data/downstream_tasks/gender_classification/mouse_gender_data",
                                        "chromosome_number": 40,
                                        "chrY_name": "chrY",
                                        "chrX_name": 'chrX',
                                        "sample_column": "strain_name",
                                        "label_column": "gender"
                                    }
                                }

        self.set_seed(seed)


    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def __iter__(self):
        # read the data once per worker (not to share h5py file object between workers)

        self.species2sampler = {}
        species2size = {}
        
        for species in self.species2metadata:
            self.species2sampler[species] = SpeciesSampler(**self.species2metadata[species], **self.common_params)

            species2size[species] = self.species2sampler[species].size
        
                        
        total_samples = sum(species2size.values())
        # self.species2prob = {species: species2size[species] / total_samples for species in species2size}
        self.species2prob = {species: 1 / len(species2size) for species in species2size}
        logger.debug(f"Species probabilities: {self.species2prob}")


        logger.debug(f"Species probabilities: {self.species2prob}")

        species, probs = zip(*self.species2prob.items())

        n_iters = 0
        while True:
            if self.max_n_samples is not None:
                if n_iters > self.max_n_samples:
                    break
            n_iters += 1
            
            # randomly select a species and its sample_id

            sampled_species = np.random.choice(species, p=probs)

            logger.debug(f"SAMPLED SPECIES: {sampled_species}")


            labels, sample_id, sampled_chrs, chunks, \
                full_sample_has_y_chr, has_y_chr_sampled, has_x_chr_sampled = self.species2sampler[sampled_species].get_species_chunk(chunk_size=self.chunk_size, n_chunks=self.n_chunks)


            # assert (labels == 0 and full_sample_has_y_chr) or (labels == 1 and not full_sample_has_y_chr), 'Inconsistent chromosome set!'

            yield {
                'labels': labels,
                'species': sampled_species,
                'sample_id': sample_id,
                'sampled_chromosomes': sampled_chrs,
                'chunks': chunks,
                'full_sample_has_y_chr': full_sample_has_y_chr,
                'has_y_chr_sampled': has_y_chr_sampled,
                'has_x_chr_sampled': has_x_chr_sampled
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

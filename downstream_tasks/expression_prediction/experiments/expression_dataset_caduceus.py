import datetime
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional

import pandas as pd
import os
import numpy as np
import hashlib

import logging
from pysam import FastaFile
import gc
import pickle
import h5py
import tqdm
import tempfile
import sys
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from downstream_tasks.expression_prediction.datasets.src.utils import convert_fm_relative_path_to_absolute_path
import pickle

class ExpressionDataset(Dataset):
    def __init__(
        self,
        gen_tokenizer,
        targets_path: str,
        text_data_path: str,
        genome: str,
        forward_intervals_path: str = None,
        reverse_intervals_path: str = None,
        loglevel: int = logging.WARNING,
        seed: int = 42,
        num_before: int = 2000,
        gen_max_seq_len: int = 4000,
        transform_targets_tpm=None,
        tpm : str = "",
        hash_prefix = None,
        n_keys: Optional[int] = None,
        token_len_for_fetch: int = 8,
        fraction_of_cell_type_specific_tpm_samples: float = 0,
        cell_type_specific_samples_path: str = None,
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=loglevel)
        # self.logger.info("Initializing dataset")

        assert sys.version_info >= (3, 8), "Python version must be 3.8 or higher" # we use dicts and realay on order of keys

        if isinstance(gen_tokenizer, str):
            self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_tokenizer, trust_remote_code=True)
        else:
            self.gen_tokenizer = gen_tokenizer

        # Len of token in bp used to compute fetch from the genome
        self.token_len_for_fetch = token_len_for_fetch

        self.gen_max_seq_len = gen_max_seq_len
        self.genome = genome

        self.seed = seed
        np.random.seed(self.seed)
        
        self.tpm = tpm
        self.targets_path = targets_path

        self.num_before = num_before
        self.transform_targets_tpm = transform_targets_tpm

        # read list of intervals (a.k.a. genes associated with intervals)
        assert forward_intervals_path is not None or reverse_intervals_path is not None, "Either forward_intervals_path or reverse_intervals_path must be provided"
        self.intervals_hash = str(forward_intervals_path) + str(reverse_intervals_path)
        if hash_prefix is None:
            self.hash_prefix = os.path.dirname(forward_intervals_path) if forward_intervals_path is not None else os.path.dirname(reverse_intervals_path)
            self.hash_prefix = os.path.join(self.hash_prefix, "dataset_hash")
        else:
            self.hash_prefix = hash_prefix

        self.read_paths()

        forward_genes = pd.read_csv(forward_intervals_path, sep = '\t') if forward_intervals_path is not None else pd.DataFrame()
        assert not "strand" in forward_genes.columns.values, "forward_intervals_path must not contain strand column"
        forward_genes["strand"] = "+"
        reverse_genes = pd.read_csv(reverse_intervals_path, sep = '\t') if reverse_intervals_path is not None else pd.DataFrame()
        assert not "strand" in reverse_genes.columns.values, "reverse_intervals_path must not contain strand column"
        reverse_genes["strand"] = "-"
        self.genes = pd.concat([forward_genes, reverse_genes])

        if n_keys is None:
            n_keys = len(self.paths.keys())
        self.n_keys = n_keys

        # Split tracks into chunks; if we have multiple datasets, we need them to have equal chunk lengths (a.k.a n_keys)
        self.n_cell_chunks = ((len(self.paths.keys()) - 1) // n_keys) + 1
        self.all_keys = list(self.paths.keys())
        self.selected_keys_chunks = []
        for i in range(self.n_cell_chunks):
            start_idx = i * n_keys
            end_idx = min((i + 1) * self.n_keys, len(self.all_keys))
            self.selected_keys_chunks.append(self.all_keys[start_idx:end_idx])

        if fraction_of_cell_type_specific_tpm_samples == 0:
            cell_type_specific_samples_path = None
        else:
            assert cell_type_specific_samples_path is not None, "cell_type_specific_samples_path must be provided if fraction_of_cell_type_specific_tpm_samples is not 0"
            
        if cell_type_specific_samples_path is not None:
            self.N_cell_type_specific_samples = max(1, int(self.n_keys * fraction_of_cell_type_specific_tpm_samples))            
            self.cell_type_specific_samples = pd.read_csv(cell_type_specific_samples_path)
            self.cell_type_specific_samples.query("cell_id in @self.all_keys", inplace=True)
            self.cell_type_specific_samples.query("gene_id in @self.genes['gene_id'].values", inplace=True)
            self.logger.debug(f"Found {len(self.cell_type_specific_samples)} cell-type-specific samples")
            if len(self.cell_type_specific_samples) == 0: # this may happen, for example, if cell type specific samples are for another species
                self.logger.warning(f"No cell-type-specific samples found for {self.targets_path} in {cell_type_specific_samples_path}")
                self.cell_type_specific_samples = None
                self.N_cell_type_specific_samples = 0
            else:
                self.cell_type_specific_samples = self.cell_type_specific_samples.groupby("gene_id")["cell_id"].apply(list).to_dict() # gene_id -> list of cell_ids
                self.n_cell_chunks = 1
        else:
            self.N_cell_type_specific_samples = 0

        self.sequences = FastaFile(self.genome)

        # Получаем путь для токенов
        self.h5_cache_path = self.get_hash_path() + ".h5"

        # Предварительно токенизируем последовательности, если кэш не существует
        if os.path.exists(self.h5_cache_path):
            self.h5_cache = h5py.File(self.h5_cache_path, "r")
        else:
            self.precompute_tokenization()

        # Read description
        with open(text_data_path, "rb") as f:
            self.desc_data = pickle.load(f)
                
        # Read tpms if tpm is not None      
        if self.tpm:
            assert all(self.paths[k][1] is not None for k in self.paths), "TPM paths are not set for some of the keys"
            tpm_hash_path = self.get_tpm_hash_path()
            if os.path.exists(tpm_hash_path):
                self.logger.debug(f"Loading tpm cache from {tpm_hash_path}")
                self.tpm_lookup = pickle.load(open(tpm_hash_path, "rb"))
                assert len(self.tpm_lookup) == len(self.paths), "Number of tpm cache and paths are not the same"
                assert all(key in self.tpm_lookup for key in self.paths.keys()), "All keys in paths must be in tpm cache"
                for key, value in self.tpm_lookup.items():
                    assert pd.isna(value).sum().sum()==0, f"TPM cache contains NaN values for {key}"
            else:
                tpm_lookup_temp = {}
                for key, (bw_paths, tpm_path) in tqdm.tqdm(self.paths.items()):
                    self.logger.debug(f"Reading tpm from {tpm_path}")
                    tpm = pd.read_csv(tpm_path, dtype=np.float32)
                    tpm_lookup_temp[key] = tpm.T.set_index(tpm.columns)
                self.tpm_lookup = tpm_lookup_temp
                pickle.dump(self.tpm_lookup, open(tpm_hash_path, "wb"))

        # Инициализируем valid_indices
        self.valid_indices = []
        self._compute_valid_indices()         

    # Вычисляем список валидных индексов
    def _compute_valid_indices(self):
        self.logger.debug("Computing valid indices...")
        valid_indices = []
        for idx in range(len(self.genes)):
            gene_id = self.genes.iloc[idx]['gene_id']
            
            # Проверяем TPM значения
            has_tpm_data = False
            for key in self.paths.keys():
                if gene_id in self.tpm_lookup[key].index:
                    has_tpm_data = True
                    break
                
            # Добавляем индекс только если есть TPM данные
            if has_tpm_data:
                valid_indices.append(idx)
                
        self.valid_indices = valid_indices
        self.logger.info(f"Found {len(self.valid_indices)} valid samples out of {len(self.genes)}")

    def get_hash_path(self):
        m = hashlib.blake2b(digest_size=8)
        input_strings = []
        
        input_str = str('tokens')
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = str(self.intervals_hash)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = str(self.genome)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = str(self.num_before)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        if self.token_len_for_fetch != 8: # 8 was default in first version of the dataset; TODO: remove at some point
            input_str = str(self.token_len_for_fetch)
            m.update(input_str.encode("utf-8"))
            input_strings.append(input_str)
            
        self.logger.debug(f"Hash inputs: {input_strings}")
        self.logger.debug(f"constructed hash suffix: {m.hexdigest()}")
        hash_suffix = m.hexdigest() 
        hash_path = str(self.hash_prefix) + ".caduceus." + hash_suffix
        return hash_path

    
    def get_tpm_hash_path(self):
        # Используем hash_prefix напрямую для TPM кэша
        return self.hash_prefix + ".caduceus.tpm.cache"
    
    def get_num_keys(self):
        return len(self.paths.keys())
        
    def read_paths(self):
        self.paths = {}
        self.logger.info(f"Reading paths from {self.targets_path}")
        df = pd.read_csv(self.targets_path)
        
        self.dataset_description = df.iloc[0]['dataset_description']

        assert not df["id"].duplicated().any(), "Found duplicated id in targets_path"

        self.paths = {k:[{"+": None, "-": None}] for ind,k in enumerate(df["id"])}


        if self.tpm:
            tpm_colname = self.tpm
            assert not pd.isna(df[tpm_colname]).any(), f"{tpm_colname} is NaN for some targets"
            tpm_paths = df[tpm_colname].apply(lambda x: convert_fm_relative_path_to_absolute_path(x, self.targets_path)).values
            for ind,k in enumerate(df["id"]):
                self.paths[k].append(tpm_paths[ind])
        else:
            for ind,k in enumerate(df["id"]):
                self.paths[k].append(None)

    def precompute_tokenization(self):
        self.logger.info(f"Precomputing tokenization to {self.h5_cache_path}")
        temp_path = f"{self.h5_cache_path}.{os.getpid()}.temp"
        
        try:
            with h5py.File(temp_path, "w") as h5f:
                pbar = tqdm.tqdm(total=len(self.genes), desc="Tokenizing sequences")
                for idx in range(len(self.genes)):
                    gene_id = self.genes.iloc[idx]['gene_id']
                    start_gene, end_gene, chrom, strand, tokens = self.tokenize_genome(idx)
                    
                    gene_group = h5f.create_group(gene_id)
                    
                    gene_group.create_dataset('input_ids', data=tokens.astype(np.int32))
                    gene_group.create_dataset('start', data=start_gene.astype(np.int64))
                    gene_group.create_dataset('end', data=end_gene.astype(np.int64))
                    gene_group.create_dataset('chrom', data=chrom.encode('utf-8'))
                    gene_group.attrs['strand'] = strand.encode('utf-8')
                    
                    if idx % 100 == 0:  
                        h5f.flush()
                    
                    pbar.update(1)
                    
                pbar.close()
                h5f.flush()
            
            os.rename(temp_path, self.h5_cache_path)
            self.h5_cache = h5py.File(self.h5_cache_path, "r")
            
        except Exception as e:
            self.logger.error(f"Error creating cache: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    # tokenize genomic sequence
    def tokenize_genome(self, i):
        row = self.genes.iloc[i]  
        chrom = row["chromosome"] 
        start = row["TSS"]
        end = row["TES"] 
        strand = row["strand"]
        
        # Инициализируем переменные по умолчанию
        start_gene = start
        end_gene = end
        sequence = ""
        tokens = np.array([], dtype=np.int32)
        
        if self.num_before > 0: 
            try:
                chrom_length = self.sequences.get_reference_length(chrom)
                start_gene = max(start - self.num_before * self.token_len_for_fetch, 0)
                end_gene = min(start + self.num_before * self.token_len_for_fetch, chrom_length)
                sequence = self.sequences.fetch(chrom, start_gene, end_gene).upper()
                
                encoded_sequence = self.gen_tokenizer.encode_plus(
                    sequence,
                    max_length=self.gen_max_seq_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                tokens = encoded_sequence['input_ids'].squeeze(0).numpy().astype(np.int32)
                #mapping = encoded_sequence['offset_mapping']  # TODO: use it
                if len(tokens) < self.num_before:
                    self.logger.warning(f"Trying to tokenize seq before TSS, but it's too short: {len(tokens)} < {self.num_before}; {chrom}: {start}-{end} ({strand})")
                    
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
                print(e.__traceback__)
                # В случае ошибки используем значения по умолчанию
                start_gene = start
                end_gene = end
                sequence = ""
                tokens = np.array([], dtype=np.int32)
        else:
            # Если num_before <= 0, используем значения по умолчанию
            start_gene = start
            end_gene = end
            sequence = ""
            tokens = np.array([], dtype=np.int32)

        return start_gene, end_gene, chrom, strand, tokens


    def __len__(self):
        return len(self.valid_indices) * self.n_cell_chunks

    def __getitem__(self, idx):
        # Преобразуем индекс в исходный индекс гена
        gene_idx = idx // self.n_cell_chunks
        original_idx = self.valid_indices[gene_idx]
        
        gene_id = self.genes.iloc[original_idx]['gene_id']
        # self.logger.debug(f"idx: {idx}, gene_id: {gene_id}")
        gene_group = self.h5_cache[gene_id]

        # Get selected keys for current chunk
        if self.N_cell_type_specific_samples > 0:
            if gene_id in self.cell_type_specific_samples:
                _cell_type_specific_samples = self.cell_type_specific_samples[gene_id]
                if len(_cell_type_specific_samples) > self.N_cell_type_specific_samples:
                    # subsample cell-type-specific samples
                    _cell_type_specific_samples = np.random.choice(_cell_type_specific_samples,
                                                self.N_cell_type_specific_samples,
                                                replace=False)
                    
                # self.logger.debug(f"N of _cell_type_specific_samples: {len(_cell_type_specific_samples)}")
                if len(_cell_type_specific_samples) < self.n_keys: # add random non-cell-type-specific samples
                    _not_cell_type_specific_samples = [key for key in self.all_keys if not key in _cell_type_specific_samples]
                    assert len(_not_cell_type_specific_samples) + len(_cell_type_specific_samples) == len(self.all_keys)
                    selected_keys = np.random.choice(_not_cell_type_specific_samples,
                                                self.n_keys - len(_cell_type_specific_samples),
                                                replace=False)
                    selected_keys = np.concatenate([_cell_type_specific_samples, selected_keys])
                elif len(_cell_type_specific_samples) > self.n_keys: # choose n_keys cell-type-specific samples
                    raise ValueError(f"N of cell-type-specific samples for {gene_id} is greater than n_keys: {len(_cell_type_specific_samples)} > {self.n_keys}")                    
                else:
                    selected_keys = _cell_type_specific_samples
            else:   # simply choice cell chunk randomly from available chunks
                # self.logger.debug(f"No cell-type-specific samples for {gene_id}")
                chunk_idx = np.random.choice(len(self.selected_keys_chunks))
                selected_keys = self.selected_keys_chunks[chunk_idx]
        else:
            chunk_idx = idx % self.n_cell_chunks
            selected_keys = self.selected_keys_chunks[chunk_idx]

        # self.logger.debug(f"selected_keys: {selected_keys}")
        
        input_ids = np.array(gene_group['input_ids'])
        start = np.array(gene_group['start'])
        end = np.array(gene_group['end'])
        chrom = gene_group['chrom'][()].decode('utf-8')
        strand = gene_group.attrs['strand']

        # sanity check
        assert strand == self.genes.iloc[original_idx]['strand']
        assert chrom == self.genes.iloc[original_idx]['chromosome']

        l = min(len(input_ids), self.gen_max_seq_len)

        features = {
            "input_ids": torch.tensor(input_ids[:l], dtype=torch.int32),
            "chrom": chrom,
            "gene_id": np.array([gene_id] * self.n_keys),
            "name": self.genes.iloc[original_idx]['gene_name'],
        }

        features["labels"] = torch.zeros((l, self.n_keys), dtype=torch.float)
        features["labels_mask"] = torch.zeros((l, self.n_keys), dtype=torch.bool)

        features["reverse"] = 0 if strand == "+" else 1
        features["start"] = start
        features["end"] = end

        # Получаем TPM значения
        tpm_values = np.full(self.n_keys, np.nan, dtype=np.float32)
        if self.tpm:
            for i, key in enumerate(selected_keys):
                if gene_id in self.tpm_lookup[key].index:
                    tpm_values[i] = self.tpm_lookup[key].loc[gene_id].iloc[0]          
            if self.transform_targets_tpm is not None:
                tpm_values = self.transform_targets_tpm(tpm_values)
            if np.all(np.isnan(tpm_values)):
                raise ValueError(f"All TPM values are NaN for {gene_id}")
            # Присваиваем TPM значения только в последнюю позицию последовательности
            features["labels"][-1, :] = torch.from_numpy(tpm_values)
            features["labels_mask"][-1, :] = True

        # Получаем desc_vectors только для текущего чанка
        desc_vectors_list = []
        for key in selected_keys:
            if key not in self.desc_data:
                raise KeyError(f"Track ID '{key}' not found in desc_data")
            desc_vec = self.desc_data[key]
            desc_vectors_list.append(desc_vec)

        # Дополняем desc_vectors нулями до n_keys
        if type(desc_vectors_list[0]) == int:
            desc_vectors = np.zeros((self.n_keys, 1), dtype=np.int32)
        else:
            desc_vectors = np.zeros((self.n_keys, len(desc_vectors_list[0])), dtype=np.float32)

        for i, vec in enumerate(desc_vectors_list):
            desc_vectors[i] = vec

        valid_tpm_count = np.count_nonzero(~np.isnan(tpm_values))
        features["dataset_description"] = [self.dataset_description] * valid_tpm_count
        if type(desc_vectors_list[0]) == int:
            features["desc_vectors"] = torch.tensor(desc_vectors, dtype=torch.int32)
        else:   
            features["desc_vectors"] = torch.tensor(desc_vectors, dtype=torch.float)
        features["selected_keys"] = selected_keys
        return features

    def __del__(self):
        if hasattr(self, 'sequences'):
            self.sequences.close()
        if hasattr(self, 'h5_cache'):
            self.h5_cache.close()

    # return main info about dataset for logging
    def describe(self):
        result = f"ExpressionDataset(n_genes={len(self.valid_indices)}, n_cell_types={len(self.paths.keys())}, n_chunks={self.n_cell_chunks}, tpm={self.tpm}"
        result += f", N_cell_type_specific_samples={self.N_cell_type_specific_samples}"
        if hasattr(self, 'dataset_description'):
            result += f", dataset_description={self.dataset_description}"
        result += ")"
        return result

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if hasattr(ds, 'open_files'):
                ds.open_files()
    else:
        if hasattr(dataset, 'open_files'):
            dataset.open_files()

class logtransform():
    def __init__(self, pseudocount=0.01):
        self.pseudocount = pseudocount
        self.rounddigits = int(abs(np.log(pseudocount)))

    def __call__(self, x):
        return np.log(x + self.pseudocount)

    def reverse(self, x):
        return np.round(np.exp(x) - self.pseudocount, self.rounddigits)
import datetime
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional

import pyBigWig as bw

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
        num_before: int = 100,
        gen_max_seq_len: int = 1008,
        transform_targets_bw=None,
        transform_targets_tpm=None,
        bw : str = "",
        tpm : str = "",
        hash_prefix = None,
        n_keys: Optional[int] = None,
    ):
        """
        Args:
            bw (str): Name of the bigwig field suffix in targets_path. I.e. `bw` -> `forward_bw` and `reverse_bw`. If empty, bw will be False.
            tpm (str): Name of the tpm field in targets_path. If empty, tpm will be False.
            n_keys (int): Number of keys to split the tracks into. If None, all tracks will be used.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=loglevel)
        # self.logger.info("Initializing dataset")

        assert sys.version_info >= (3, 8), "Python version must be 3.8 or higher" # we use dicts and realay on order of keys

        if isinstance(gen_tokenizer, str):
            self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_tokenizer)
        else:
            self.gen_tokenizer = gen_tokenizer

        self.gen_max_seq_len = gen_max_seq_len
        self.genome = genome

        self.seed = seed
        np.random.seed(self.seed)
        
        self.bw = bw
        self.tpm = tpm
        self.targets_path = targets_path
        self.read_paths()

        if n_keys is None:
            n_keys = len(self.paths.keys())
        self.n_keys = n_keys

        # Split tracks into chunks; if we have multiple datasets, we need them to have equal chunk lengths (a.k.a n_keys)
        self.n_cell_chunks = ((len(self.paths.keys()) - 1) // n_keys) + 1
        all_keys = list(self.paths.keys())
        self.selected_keys_chunks = []
        for i in range(self.n_cell_chunks):
            start_idx = i * n_keys
            end_idx = min((i + 1) * n_keys, len(all_keys))
            self.selected_keys_chunks.append(all_keys[start_idx:end_idx])
        
        self.num_before = num_before
        self.transform_targets_bw = transform_targets_bw
        self.transform_targets_tpm = transform_targets_tpm

        # read list of intervals (a.k.a. genes associated with intervals)
        assert forward_intervals_path is not None or reverse_intervals_path is not None, "Either forward_intervals_path or reverse_intervals_path must be provided"
        self.intervals_hash = str(forward_intervals_path) + str(reverse_intervals_path)
        if hash_prefix is None:
            self.hash_prefix = os.path.dirname(forward_intervals_path) if forward_intervals_path is not None else os.path.dirname(reverse_intervals_path)
            self.hash_prefix = os.path.join(self.hash_prefix, "dataset_hash")
        else:
            self.hash_prefix = hash_prefix

        forward_genes = pd.read_csv(forward_intervals_path, sep = '\t') if forward_intervals_path is not None else pd.DataFrame()
        assert not "strand" in forward_genes.columns.values, "forward_intervals_path must not contain strand column"
        forward_genes["strand"] = "+"
        reverse_genes = pd.read_csv(reverse_intervals_path, sep = '\t') if reverse_intervals_path is not None else pd.DataFrame()
        assert not "strand" in reverse_genes.columns.values, "reverse_intervals_path must not contain strand column"
        reverse_genes["strand"] = "-"
        self.genes = pd.concat([forward_genes, reverse_genes])
        
        self.files_opened = False

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

        # Read bw if bw is not None
        if self.bw:
            self.signals_cache_path = self.get_signals_hash_path() + ".h5"
            if os.path.exists(self.signals_cache_path):
                self.signals_cache = h5py.File(self.signals_cache_path, "r")
            else:
                self.precompute_signals()
                
        # Read tpms if tpm is not None      
        if self.tpm:
            assert all(self.paths[k][1] is not None for k in self.paths), "TPM paths are not set for some of the keys"
            tpm_hash_path = self.get_tpm_hash_path()
            if os.path.exists(tpm_hash_path):
                self.tpm_lookup = pickle.load(open(tpm_hash_path, "rb"))
                assert len(self.tpm_lookup) == len(self.paths), "Number of tpm cache and paths are not the same"
                assert all(key in self.tpm_lookup for key in self.paths.keys()), "All keys in paths must be in tpm cache"
                for key, value in self.tpm_lookup.items():
                    assert pd.isna(value).sum().sum()==0, f"TPM cache contains NaN values for {key}"
            else:
                self.tpm_cache = {}
                for key, (bw_paths, tpm_path) in self.paths.items():
                    self.logger.info(f"Reading tpm from {tpm_path}")
                    tpm = pd.read_csv(tpm_path, dtype=np.float32)
                    self.tpm_cache[key] = tpm
                self.tpm_lookup = {}
                for key, tpm_df in self.tpm_cache.items():
                    self.tpm_lookup[key] = tpm_df.T.set_index(tpm_df.columns)
                pickle.dump(self.tpm_lookup, open(tpm_hash_path, "wb"))

        # Добавляем список валидных индексов
        self.valid_indices = []
        if not self.bw:  # Вычисляем валидные индексы только если bw=False
            self._compute_valid_indices()
        else:
            # Если bw=True, все индексы валидны
            self.valid_indices = list(range(len(self.genes)))

    # Вычисляем список валидных индексов
    def _compute_valid_indices(self):
        self.logger.info("Computing valid indices...")
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
                self.valid_indices.append(idx)
                
        self.logger.info(f"Found {len(self.valid_indices)} valid samples out of {len(self.genes)}")

    def get_hash_path(self):
        m = hashlib.blake2b(digest_size=8)
        m.update(str('tokens').encode("utf-8"))
        m.update(str('test2').encode("utf-8"))
        m.update(str(self.intervals_hash).encode("utf-8"))
        m.update(str(self.genome).encode("utf-8"))
        m.update(str(self.num_before).encode("utf-8"))
        hash_suffix = m.hexdigest()
        hash_path = str(self.hash_prefix) + "." + hash_suffix
        return hash_path

    def get_signals_hash_path(self):
        m = hashlib.blake2b(digest_size=8)
        m.update(str('signals').encode("utf-8"))
        m.update(str('test2').encode("utf-8"))
        m.update(str(self.intervals_hash).encode("utf-8"))
        m.update(str(self.targets_path).encode("utf-8"))
        m.update(str(self.genome).encode("utf-8"))
        m.update(str(self.num_before).encode("utf-8"))
        m.update(str(self.gen_max_seq_len).encode("utf-8"))
        target_ids = "".join(sorted(list(self.paths.keys())))
        m.update(str(target_ids).encode("utf-8"))
        hash_suffix = m.hexdigest()
        return str(self.hash_prefix) + ".signal." + hash_suffix
    
    def get_tpm_hash_path(self):
        signals_hash_path = self.get_signals_hash_path()
        return self.hash_prefix + ".tpm." + signals_hash_path[len(self.hash_prefix) + len(".signal."):]
    
    def get_num_keys(self):
        return len(self.paths.keys())
        
    def read_paths(self):
        self.paths = {}
        self.logger.info(f"Reading paths from {self.targets_path}")
        df = pd.read_csv(self.targets_path)

        self.dataset_description = df.iloc[0]['dataset_description']

        assert not df["id"].duplicated().any(), "Found duplicated id in targets_path"

        if self.bw:
            forward_colname = "forward_"+self.bw
            reverse_colname = "reverse_"+self.bw
            assert not pd.isna(df[forward_colname]).any(), f"{forward_colname} is NaN for some targets"
            assert not pd.isna(df[reverse_colname]).any(), f"{reverse_colname} is NaN for some targets"
            forward_paths = df[forward_colname].apply(lambda x: convert_fm_relative_path_to_absolute_path(x, self.targets_path)).values
            reverse_paths = df[reverse_colname].apply(lambda x: convert_fm_relative_path_to_absolute_path(x, self.targets_path)).values
            self.paths = {k:[{"+": forward_paths[ind], "-": reverse_paths[ind]}] for ind,k in enumerate(df["id"])}
        else:
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

        self.files_opened = False

    def precompute_tokenization(self):
        self.logger.info(f"Precomputing tokenization to {self.h5_cache_path}")
        temp_path = f"{self.h5_cache_path}.{os.getpid()}.temp"
        
        try:
            with h5py.File(temp_path, "w") as h5f:
                pbar = tqdm.tqdm(total=len(self.genes), desc="Tokenizing sequences")
                for idx in range(len(self.genes)):
                    gene_id = self.genes.iloc[idx]['gene_id']
                    _, tokens_df = self.tokenize_genome(idx)
                    
                    gene_group = h5f.create_group(gene_id)
                    
                    gene_group.create_dataset('input_ids', data=tokens_df["token_id"].values.astype(np.int32))
                    gene_group.create_dataset('starts', data=tokens_df["start"].values.astype(np.int64))
                    gene_group.create_dataset('ends', data=tokens_df["end"].values.astype(np.int64))
                    if len(tokens_df["chrom"]) > 0:
                        gene_group.create_dataset('chrom', data=tokens_df["chrom"].iloc[0].encode('utf-8'))
                    else:
                        self.logger.warning(f"No chromosomes found for gene {gene_id}:\n {self.genes.iloc[idx]}")
                        raise ValueError(f"No chromosomes found for gene {gene_id}:\n {self.genes.iloc[idx]}. \n Possible reason - genome mismatch.")
                    gene_group.attrs['strand'] = self.genes.iloc[idx]['strand'].encode('utf-8')
                    
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
    
    # Check if bigwig file is consistent with genome
    def check_bw_genome_consistency(self, bw_handler, sequences):
        # get reference names and lengths from sequences
        fasta_ref_lengths = {k:v for k, v in zip(sequences.references, sequences.lengths)}
        
        # get reference names and lengths from bw_handler
        bw_ref_lengths = {k:v for k, v in bw_handler.chroms().items()}

        # intersect keys of fasta_ref_lengths and bw_ref_lengths
        common_refs = set(fasta_ref_lengths.keys()) & set(bw_ref_lengths.keys())
        assert len(common_refs) > 0, f"No common references found in genome and bigwig file. Genome: {fasta_ref_lengths.keys()}, Bigwig: {bw_ref_lengths.keys()}. Genome mismatch?"

        # check if lengths are the same
        for ref in common_refs:
            if fasta_ref_lengths[ref] != bw_ref_lengths[ref]:
                raise ValueError(f"Length of {ref} in genome and bigwig file are different. Ref: {ref}, Genome: {fasta_ref_lengths[ref]}, Bigwig: {bw_ref_lengths[ref]}")
    
    def open_files(self):
        if self.bw:
            self.bigWigHandlers = {}
            for k, (v1, v2) in self.paths.items():
                try:
                    self.bigWigHandlers[k] = {strand: bw.open(path) for strand, path in v1.items()}
                    for bw_handler in self.bigWigHandlers[k].values():
                        self.check_bw_genome_consistency(bw_handler, self.sequences)
                except Exception as e:
                    self.logger.error(f"Error opening bigwig file {v1}")
                    print(e.__traceback__.format_exc())
            self.files_opened = True

    def reverse_complement(self, sequence):
        complement = str.maketrans('ACGTN', 'TGCAN')
        return sequence.translate(complement)[::-1]

    # tokenize genomic sequence
    def tokenize_genome(self, i):
        row = self.genes.iloc[i]  
        chrom = row["chromosome"] 
        start = row["TSS"]
        end = row["TES"] 
        strand = row["strand"]
        reverse = 0 if strand == "+" else 1

        # if (reverse == 0): # forward strand
        #     try:
        #         sequence = self.sequences.fetch(chrom, max(start - self.num_before * 9, 0), start).upper()
        #     except ValueError as e:
        #         self.logger.error(f"Error sequence {i}")
        #         print(e.__traceback__)
        # else: # reverse strand
        #     chrom_length = self.sequences.get_reference_length(chrom)
        #     try:
        #         sequence = self.sequences.fetch(chrom, start, min(start + self.num_before * 9, chrom_length)).upper()
        #         sequence = self.reverse_complement(sequence)
        #     except ValueError as e:
        #         self.logger.error(f"Error sequence {i}")
        #         print(e.__traceback__)
            
        # encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        # encoded_sequence['input_ids'] = encoded_sequence['input_ids'][1:-1]
        # encoded_sequence['offset_mapping'] = encoded_sequence['offset_mapping'][1:-1]
        # tokens_before = encoded_sequence['input_ids'][-self.num_before:]
        # mapping = encoded_sequence['offset_mapping'][-self.num_before:]
        
        # token_lengths = []
        # for i, (start_i, end_i) in enumerate(mapping):
        #     token_id = tokens_before[i]
        #     if (token_id == 5):
        #         if i > 0:
        #             length = end_i - mapping[i-1][1] 
        #         else:
        #             length = end_i
        #     else:
        #         length = end_i - start_i  
        #     token = self.gen_tokenizer.decode([token_id])  
        #     token_lengths.append((token_id, token, length))

        token_lengths = []
    
        if reverse == 0:
            start_gene = start - sum(t[2] for t in token_lengths)
        else:
            start_gene = end
    
        if reverse == 0:
            try:
                sequence = self.sequences.fetch(chrom, start, start + 9 * 1500).upper()
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
                print(e.__traceback__)
        else:
            try:
                sequence = self.sequences.fetch(chrom, start - 9 * 1500, start).upper()
                sequence = self.reverse_complement(sequence)
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
                print(e.__traceback__)
        
        encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        tokens_before = encoded_sequence['input_ids'][1:-1]
        mapping = encoded_sequence['offset_mapping'][1:-1]
       
        for i, (start_i, end_i) in enumerate(mapping):
            token_id = tokens_before[i]
            if (token_id == 5):
                if i > 0:
                    length = end_i - mapping[i-1][1] 
                else:
                    length = end_i
            else:
                length = end_i - start_i 
            token = self.gen_tokenizer.decode([token_id])  
            token_lengths.append((token_id, token, length))
            
        if reverse == 1: # reverse strand
            token_lengths.reverse()
        token_lengths_df = pd.DataFrame(token_lengths, columns=['token_id', 'token', 'length'])
        token_lengths_df['start'] = token_lengths_df['length'].cumsum().shift(fill_value=0) + start_gene 
        token_lengths_df['end'] = token_lengths_df['start'] + token_lengths_df['length']
        token_lengths_df['chrom'] = chrom
        if reverse == 1: # reverse strand
            token_lengths_df = token_lengths_df[::-1].reset_index(drop=True)
        token_lengths_df = token_lengths_df[50:]
        return start_gene, token_lengths_df

    def process_region_signals(self, bw, chrom, starts, ends, l, strand):

        reverse = 0 if strand == "+" else 1

        signals = np.zeros(l, dtype=np.float32)
        
        # Определяем границы всего региона
        if reverse == 0:
            region_start = int(starts[0])
            region_end = int(ends[-1])
        else:
            region_start = int(starts[-1])
            region_end = int(ends[0])
        
        if region_start >= region_end:
            return signals
            
        try:
            # Получаем все интервалы сразу для всего региона
            intervals = bw.intervals(chrom, region_start, region_end)
            
            if not intervals:
                return signals
                
            # Создаем массив для хранения значений на каждую позицию
            region_size = region_end - region_start
            position_values = np.zeros(region_size, dtype=np.float32)
            
            # Заполняем массив значениями из интервалов
            for interval_start, interval_end, value in intervals:
                # Преобразуем геномные координаты в индексы массива
                rel_start = max(0, interval_start - region_start)
                rel_end = min(region_size, interval_end - region_start)
                
                if rel_start < rel_end:
                    position_values[rel_start:rel_end] = value
            
            # Вычисляем суммы для каждого токена
            for j in range(l):
                token_start = max(0, int(starts[j]) - region_start)
                token_end = min(region_size, int(ends[j]) - region_start)
                
                if token_start < token_end:
                    signals[j] = np.sum(position_values[token_start:token_end])
                    
        except Exception as e:
            self.logger.error(f"Error processing signals for {chrom}:{region_start}-{region_end}: {e}")
            
        return signals

    def precompute_signals(self):
        self.logger.info(f"Precomputing signals to {self.signals_cache_path}")
        temp_path = f"{self.signals_cache_path}.{os.getpid()}.temp"
        
        try:
            if not self.files_opened:
                self.open_files()

            with h5py.File(temp_path, "w") as h5f:
                pbar = tqdm.tqdm(total=len(self.genes), desc="Computing signals")
                
                for idx in range(len(self.genes)):
                    gene_id = self.genes.iloc[idx]['gene_id']
                    gene_group = self.h5_cache[gene_id]
                    
                    input_ids = np.array(gene_group['input_ids'])
                    starts = np.array(gene_group['starts'])
                    ends = np.array(gene_group['ends'])
                    chrom = gene_group['chrom'][()].decode('utf-8')
                    strand = gene_group.attrs['strand']

                    # sanity check
                    assert strand == self.genes.iloc[idx]['strand']
                    assert chrom[0] == self.genes.iloc[idx]['chromosome'][0]

                    l = min(len(input_ids), self.gen_max_seq_len)
                    
                    bigwig_signals = np.zeros((l, len(self.bigWigHandlers)), dtype=np.float32)
                    
                    for i, (key, bw) in enumerate(self.bigWigHandlers.items()):
                        track_signals = self.process_region_signals(bw[strand], chrom, starts[:l], ends[:l], l, strand)
                        bigwig_signals[:, i] = track_signals
                    
                    signals_group = h5f.create_group(gene_id)
                    signals_group.create_dataset('signals', data=bigwig_signals)
                    
                    if idx % 100 == 0: 
                        h5f.flush()
                    
                    pbar.update(1)
                
                pbar.close()
                h5f.flush()
            
            os.rename(temp_path, self.signals_cache_path)
            self.signals_cache = h5py.File(self.signals_cache_path, "r")
            
        except Exception as e:
            self.logger.error(f"Error creating signals cache: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def __len__(self):
        return len(self.valid_indices) * self.n_cell_chunks

    def __getitem__(self, idx):
        # Преобразуем индекс в исходный индекс гена
        gene_idx = idx // self.n_cell_chunks
        original_idx = self.valid_indices[gene_idx]
        chunk_idx = idx % self.n_cell_chunks

        # Получаем выбранные ключи для текущего чанка
        selected_keys = self.selected_keys_chunks[chunk_idx]
        
        if not self.files_opened and self.bw:
            self.open_files()

        gene_id = self.genes.iloc[original_idx]['gene_id']
        gene_group = self.h5_cache[gene_id]
        
        input_ids = np.array(gene_group['input_ids'])
        starts = np.array(gene_group['starts'])
        ends = np.array(gene_group['ends'])
        chrom = gene_group['chrom'][()].decode('utf-8')
        strand = gene_group.attrs['strand']

        # sanity check
        assert strand == self.genes.iloc[original_idx]['strand']
        assert chrom == self.genes.iloc[original_idx]['chromosome']

        l = min(len(input_ids), self.gen_max_seq_len)

        features = {
            "input_ids": torch.tensor(input_ids[:l], dtype=torch.int32),
            "attention_mask": torch.ones(l, dtype=torch.bool),
            "token_type_ids": torch.zeros(l, dtype=torch.int32),
            "chrom": chrom,
            "gene_id": [gene_id] * self.n_keys,
            "name": self.genes.iloc[original_idx]['gene_name'],
        }

        if self.bw:
            # Load from cache 
            if self.signals_cache is not None:
                gene_id = self.genes.iloc[original_idx]['gene_id']
                signals_group = self.signals_cache[gene_id]
                bigwig_signals = np.array(signals_group['signals'])
            else:
                # If cache is not found, compute signals 
                bigwig_signals = np.zeros((l, len(self.bigWigHandlers)), dtype=np.float32)
                for i, (key, bw) in enumerate(self.bigWigHandlers.items()):
                    track_signals = self.process_region_signals(bw[strand], chrom, starts[:l], ends[:l], l, strand)
                    bigwig_signals[:, i] = track_signals

            chunk_signals = np.zeros((l, self.n_keys), dtype=np.float32)
            chunk_mask = np.zeros((l, self.n_keys), dtype=bool)
            
            for i, key in enumerate(selected_keys):
                if key in self.bigWigHandlers:
                    chunk_signals[:, i] = bigwig_signals[:, list(self.bigWigHandlers.keys()).index(key)]
                    chunk_mask[:, i] = True
                else:
                    raise KeyError(f"Track ID '{key}' not found in bigwig handlers")
    
            if self.transform_targets_bw is not None:
                chunk_signals = self.transform_targets_bw(chunk_signals)
        
            features["labels"] = torch.tensor(chunk_signals, dtype=torch.float)
            features["labels_mask"] = torch.tensor(chunk_mask, dtype=torch.bool)
        
        else:
            features["labels"] = torch.zeros((l, self.n_keys), dtype=torch.float)
            features["labels_mask"] = torch.zeros((l, self.n_keys), dtype=torch.bool)

        reverse = 0 if strand == "+" else 1
        if reverse == 0 :
            features["reverse"] = 0
            features["start"] = starts[0]
            features["end"] = ends[l-1]
        else:
            features["reverse"] = 1
            features["start"] = starts[l-1]
            features["end"] = ends[0]

        # Получаем TPM значения
        tpm_values = np.full(self.n_keys, np.nan, dtype=np.float32)
        if self.tpm:
            for i, key in enumerate(selected_keys):
                if gene_id in self.tpm_lookup[key].index:
                    tpm_values[i] = self.tpm_lookup[key].loc[gene_id].iloc[0]
            
            if not np.all(np.isnan(tpm_values)) and self.transform_targets_tpm is not None:
                tpm_values = self.transform_targets_tpm(tpm_values)
        
        features["tpm"] = torch.from_numpy(tpm_values)
        

        # Получаем desc_vectors только для текущего чанка
        desc_vectors_list = []
        for key in selected_keys:
            if key not in self.desc_data:
                raise KeyError(f"Track ID '{key}' not found in desc_data")
            desc_vec = self.desc_data[key]
            desc_vectors_list.append(desc_vec)

        # Дополняем desc_vectors нулями до n_keys
        desc_vectors = np.zeros((self.n_keys, len(desc_vectors_list[0])), dtype=np.float32)
        for i, vec in enumerate(desc_vectors_list):
            desc_vectors[i] = vec

        valid_tpm_count = np.count_nonzero(~np.isnan(tpm_values))
        features["dataset_description"] = [self.dataset_description] * valid_tpm_count
        features["desc_vectors"] = torch.tensor(desc_vectors, dtype=torch.float)
        features["selected_keys"] = selected_keys
        return features

    def __del__(self):
        if hasattr(self, 'sequences'):
            self.sequences.close()
        if hasattr(self, 'h5_cache'):
            self.h5_cache.close()
        if hasattr(self, 'signals_cache'):
            self.signals_cache.close()

    # return main info about dataset for logging
    def describe(self):
        return f"ExpressionDataset(n_genes={len(self.valid_indices)}, n_cell_types={len(self.paths.keys())}, n_chunks={self.n_cell_chunks}, bw={self.bw}, tpm={self.tpm})"

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
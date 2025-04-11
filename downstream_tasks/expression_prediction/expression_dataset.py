import datetime
import torch
from torch.utils.data import Dataset, ConcatDataset

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

class ExpressionDataset(Dataset):
    def __init__(
        self,
        gen_tokenizer,
        targets_path: str,
        intervals_path: str,
        text_data_path: str,
        genome: str,
        loglevel: int = logging.WARNING,
        seed: int = 42,
        num_before: int = 100,
        gen_max_seq_len: int = 1008,
        reverse: int = 0,
        transform_targets_bw=None,
        transform_targets_tpm=None,
        bw = True,
        tpm = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=loglevel)
        self.logger.info("Initializing dataset")

        if isinstance(gen_tokenizer, str):
            self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_tokenizer)
        else:
            self.gen_tokenizer = gen_tokenizer

        self.gen_max_seq_len = gen_max_seq_len
        self.reverse = reverse
        self.genome = genome
        self.intervals_path = intervals_path
        self.seed = seed
        np.random.seed(self.seed)
        self.num_before = num_before
        self.transform_targets_bw = transform_targets_bw
        self.transform_targets_tpm = transform_targets_tpm
        self.targets_path = targets_path

        self.read_paths()
        self.genes = pd.read_csv(intervals_path, sep = '\t')
        self.files_opened = False
        self.bw = bw
        self.tpm = tpm

        # Открываем Fasta один раз
        self.sequences = FastaFile(self.genome)

        # Получаем путь для токенов
        self.h5_cache_path = self.get_hash_path() + ".h5"

        # Предварительно токенизируем последовательности, если кэш не существует
        if os.path.exists(self.h5_cache_path):
            self.h5_cache = h5py.File(self.h5_cache_path, "r")
        else:
            self.precompute_tokenization()

        # Достаем описания
        with open(text_data_path, "rb") as f:
            self.desc_data = pickle.load(f)

        # Работаем с bw, если bw = True
        if self.bw:
            self.signals_cache_path = self.get_signals_hash_path() + ".h5"
            if os.path.exists(self.signals_cache_path):
                self.signals_cache = h5py.File(self.signals_cache_path, "r")
            else:
                self.precompute_signals()
                
        # Работаем с tpm, если tpm = True      
        if self.tpm:
            self.tpm_cache = {}
            for key, (v1 ,v2) in self.paths.items():
                tpm = pd.read_csv(v2)
                self.tpm_cache[key] = tpm
            self.tpm_lookup = {}
            for key, tpm_df in self.tpm_cache.items():
                self.tpm_lookup[key] = tpm_df.T.set_index(tpm_df.columns)

        # Добавляем список валидных индексов
        self.valid_indices = []
        if not self.bw:  # Вычисляем валидные индексы только если bw=False
            self._compute_valid_indices()
        else:
            # Если bw=True, все индексы валидны
            self.valid_indices = list(range(len(self.genes)))

    def _compute_valid_indices(self):
        """Предварительно вычисляем список валидных индексов"""
        self.logger.info("Computing valid indices...")
        for idx in tqdm.tqdm(range(len(self.genes)), desc="Checking samples"):
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
        m.update(str(self.intervals_path).encode("utf-8"))
        m.update(str(self.reverse).encode("utf-8"))
        m.update(str(self.genome).encode("utf-8"))
        m.update(str(self.num_before).encode("utf-8"))
        hash_suffix = m.hexdigest()
        hash_path = str(self.intervals_path) + "." + hash_suffix
        return hash_path

    def get_signals_hash_path(self):
        m = hashlib.blake2b(digest_size=8)
        m.update(str('signals').encode("utf-8"))
        m.update(str(self.intervals_path).encode("utf-8"))
        m.update(str(self.targets_path).encode("utf-8"))
        m.update(str(self.reverse).encode("utf-8"))
        m.update(str(self.genome).encode("utf-8"))
        m.update(str(self.num_before).encode("utf-8"))
        m.update(str(self.gen_max_seq_len).encode("utf-8"))  
        hash_suffix = m.hexdigest()
        return str(self.intervals_path) + ".signal." + hash_suffix
        
    def read_paths(self):
        self.paths = {} 
        df = pd.read_csv(self.targets_path)
        
        # Получаем директорию, в которой находится targets_path
        base_dir = os.path.dirname(os.path.abspath(self.targets_path))
        
        for _, row in df.iterrows():
            k = row["id"]
            
            # Обрабатываем пути для v2
            if pd.isna(row["csv"]):  
                v2 = row["CPM"]  # оставляем как есть (может быть NaN)
            else:
                v2 = row["csv"]  # оставляем как есть (может быть NaN)
                
            # Если путь существует и не NaN, делаем его абсолютным
            if not pd.isna(v2) and not str(v2).startswith('/'):
                v2 = os.path.join(base_dir, str(v2))

            # Обрабатываем пути для v1
            if self.reverse == 0:
                v1 = row["forward_bw"]  # оставляем как есть (может быть NaN)
            else:
                v1 = row["reverse_bw"]  # оставляем как есть (может быть NaN)
                
            # Если путь существует и не NaN, делаем его абсолютным
            if not pd.isna(v1) and not str(v1).startswith('/'):
                v1 = os.path.join(base_dir, str(v1))
            
            assert k not in self.paths, f"Found repeated name {k}"
            self.paths[k] = (v1, v2)  
        
        self.files_opened = False

    def precompute_tokenization(self):
        self.logger.info(f"Precomputing tokenization to {self.h5_cache_path}")
        temp_path = f"{self.h5_cache_path}.{os.getpid()}.temp"
        
        try:
            with h5py.File(temp_path, "w") as h5f:
                pbar = tqdm.tqdm(total=len(self.genes), desc="Tokenizing sequences")
                for idx in range(len(self.genes)):
                    gene_id = self.genes.iloc[idx]['gene_id']
                    start_gene, tokens_df = self.tokenize_genome(idx)
                    
                    gene_group = h5f.create_group(gene_id)
                    
                    gene_group.create_dataset('input_ids', data=tokens_df["token_id"].values.astype(np.int32))
                    gene_group.create_dataset('starts', data=tokens_df["start"].values.astype(np.int64))
                    gene_group.create_dataset('ends', data=tokens_df["end"].values.astype(np.int64))
                    gene_group.create_dataset('chrom', data=tokens_df["chrom"].iloc[0].encode('utf-8'))
                    
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

    def open_files(self):
        if self.bw:
            self.bigWigHandlers = {}
            for k, (v1, v2) in self.paths.items():
                try:
                    self.bigWigHandlers[k] = bw.open(v1)
                except Exception as e:
                    self.logger.error(f"Error opening bigwig file {v1}")
                    print(e.__traceback__)
            self.files_opened = True

    def reverse_complement(self, sequence):
        complement = str.maketrans('ACGTN', 'TGCAN')
        return sequence.translate(complement)[::-1]

    # Токенизируем последовательности
    def tokenize_genome(self, i):
        row = self.genes.iloc[i]  
        chrom = row["chromosome"] 
        start = row["TSS"]
        end = row["TES"] 
        
        if (self.reverse == 0):
            try:
                sequence = self.sequences.fetch(chrom, max(start - self.num_before * 9, 0), start).upper()
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
                print(e.__traceback__)
        else:
            chrom_length = self.sequences.get_reference_length(chrom)
            try:
                sequence = self.sequences.fetch(chrom, start, min(start + self.num_before * 9, chrom_length)).upper()
                sequence = self.reverse_complement(sequence)
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
                print(e.__traceback__)
            
        encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        encoded_sequence['input_ids'] = encoded_sequence['input_ids'][1:-1]
        encoded_sequence['offset_mapping'] = encoded_sequence['offset_mapping'][1:-1]
        tokens_before = encoded_sequence['input_ids'][-self.num_before:]
        mapping = encoded_sequence['offset_mapping'][-self.num_before:]
        
        token_lengths = []
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
    
        if self.reverse == 0:
            start_gene = start - sum(t[2] for t in token_lengths)
        else:
            start_gene = end
    
        if (self.reverse == 0):
            try:
                sequence = self.sequences.fetch(chrom, start, end).upper()
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
                print(e.__traceback__)
        else:
            try:
                sequence = self.sequences.fetch(chrom, end, start).upper()
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
            
        if self.reverse == 1:
            token_lengths.reverse()
        token_lengths_df = pd.DataFrame(token_lengths, columns=['token_id', 'token', 'length'])
        token_lengths_df['start'] = token_lengths_df['length'].cumsum().shift(fill_value=0) + start_gene 
        token_lengths_df['end'] = token_lengths_df['start'] + token_lengths_df['length']
        token_lengths_df['chrom'] = chrom
        if self.reverse == 1:
            token_lengths_df = token_lengths_df[::-1].reset_index(drop=True)
        return start_gene, token_lengths_df

    def process_region_signals(self, bw, chrom, starts, ends, l):
        signals = np.zeros(l, dtype=np.float32)
        
        # Определяем границы всего региона
        if self.reverse == 0:
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
                    
                    l = min(len(input_ids), self.gen_max_seq_len)
                    
                    bigwig_signals = np.zeros((l, len(self.bigWigHandlers)), dtype=np.float32)
                    
                    for i, (key, bw) in enumerate(self.bigWigHandlers.items()):
                        track_signals = self.process_region_signals(bw, chrom, starts[:l], ends[:l], l)
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
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Преобразуем индекс в исходный индекс гена
        original_idx = self.valid_indices[idx]
        
        if not self.files_opened and self.bw:
            self.open_files()

        gene_id = self.genes.iloc[original_idx]['gene_id']
        gene_group = self.h5_cache[gene_id]
        
        input_ids = np.array(gene_group['input_ids'])
        starts = np.array(gene_group['starts'])
        ends = np.array(gene_group['ends'])
        chrom = gene_group['chrom'][()].decode('utf-8')
                
        l = min(len(input_ids), self.gen_max_seq_len)

        features = {
            "input_ids": torch.tensor(input_ids[:l], dtype=torch.int32),
            "attention_mask": torch.ones(l, dtype=torch.bool),
            "token_type_ids": torch.zeros(l, dtype=torch.int32),
            "chrom": chrom,
            "gene_id": gene_id,
            "name": self.genes.iloc[original_idx]['gene_name'],
        }

        if self.bw:
            # Загружаем сигналы из кэша 
            if self.signals_cache is not None:
                gene_id = self.genes.iloc[original_idx]['gene_id']
                signals_group = self.signals_cache[gene_id]
                bigwig_signals = np.array(signals_group['signals'])
            else:
                # Если кэша нет, вычисляем сигналы 
                bigwig_signals = np.zeros((l, len(self.bigWigHandlers)), dtype=np.float32)
                for i, (key, bw) in enumerate(self.bigWigHandlers.items()):
                    track_signals = self.process_region_signals(bw, chrom, starts[:l], ends[:l], l)
                    bigwig_signals[:, i] = track_signals
    
            if self.transform_targets_bw is not None:
                bigwig_signals = self.transform_targets_bw(bigwig_signals)
        
            features["labels"] = torch.tensor(bigwig_signals, dtype=torch.float)
            features["labels_mask"] = torch.ones(l, dtype=torch.bool)
        
        else:
            features["labels"] = torch.zeros((l, len(self.paths)), dtype=torch.float)
            features["labels_mask"] = torch.zeros(l, dtype=torch.bool)

        if self.reverse == 0 :
            features["start"] = starts[0]
            features["end"] = ends[l-1]
        else:
            features["start"] = starts[l-1]
            features["end"] = ends[0]

        # Получаем TPM значения
        if not self.tpm:
            tpm_values = np.full(len(self.paths), np.nan, dtype=np.float32)
        else:
            tpm_values = np.full(len(self.paths), np.nan, dtype=np.float32)
            for i, key in enumerate(self.paths.keys()):
                if gene_id in self.tpm_lookup[key].index:
                    tpm_values[i] = self.tpm_lookup[key].loc[gene_id].iloc[0]
            
            if not np.all(np.isnan(tpm_values)) and self.transform_targets_tpm is not None:
                tpm_values = self.transform_targets_tpm(tpm_values)
        
        features["tpm"] = torch.from_numpy(tpm_values)

        # Получаем desc_vectors
        desc_vectors_list = []
        for key in self.paths.keys():
            if key not in self.desc_data:
                raise KeyError(f"Track ID '{key}' not found in desc_data")
            desc_vec = self.desc_data[key]
            desc_vectors_list.append(desc_vec)

        desc_vectors = np.stack(desc_vectors_list, axis=0)
        features["desc_vectors"] = torch.tensor(desc_vectors, dtype=torch.float)

        return features

    def __del__(self):
        if hasattr(self, 'sequences'):
            self.sequences.close()
        if hasattr(self, 'h5_cache'):
            self.h5_cache.close()
        if hasattr(self, 'signals_cache'):
            self.signals_cache.close()

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
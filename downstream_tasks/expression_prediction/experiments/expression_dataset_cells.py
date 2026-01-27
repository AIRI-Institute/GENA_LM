import datetime
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, Dict, List, Tuple, Any

import pyBigWig as bw
import pickle

import pandas as pd
import os
import numpy as np
import hashlib

import logging
from pysam import FastaFile
import gc
import h5py
import tqdm
import sys
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from downstream_tasks.expression_prediction.datasets.src.utils import convert_fm_relative_path_to_absolute_path

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
        token_len_for_fetch: int = 8,
        fraction_of_cell_type_specific_tpm_samples: float = 0,
        cell_type_specific_samples_path: str = None,
        norm_bw = False
    ):
        """
        Основные отличия от исходной версии:
        - __getitem__ возвращает «батч по ключам» (n_keys, l) вместо одной последовательности.
        - labels теперь (n_keys, l).
        - open_files строит быстрый mapping key->col.
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=loglevel)

        assert sys.version_info >= (3, 8), "Python 3.8+ required"

        if isinstance(gen_tokenizer, str):
            # при желании можно убрать trust_remote_code=True для безопасности
            self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_tokenizer, trust_remote_code=True)
        else:
            self.gen_tokenizer = gen_tokenizer

        # Len of token in bp used to compute fetch from the genome
        self.token_len_for_fetch = token_len_for_fetch

        self.gen_max_seq_len = gen_max_seq_len
        self.genome = genome

        self.seed = seed
        np.random.seed(self.seed)
        
        self.bw = bw
        self.tpm = tpm
        self.norm_bw = norm_bw 
        
        self.targets_path = targets_path

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

        self.read_paths()

        forward_genes = pd.read_csv(forward_intervals_path, sep=None, engine="python") if forward_intervals_path is not None else pd.DataFrame()
        forward_genes["strand"] = "+"
        reverse_genes = pd.read_csv(reverse_intervals_path, sep=None, engine="python") if reverse_intervals_path is not None else pd.DataFrame()
        reverse_genes["strand"] = "-"
        self.genes = pd.concat([forward_genes, reverse_genes], ignore_index=True)

        if n_keys is None:
            n_keys = len(self.paths.keys())
        self.n_keys = n_keys

        # Split tracks into chunks; if we have multiple datasets, we need them to have equal chunk lengths (a.k.a n_keys)
        self.n_cell_chunks = ((len(self.paths.keys()) - 1) // n_keys) + 1
        self.all_keys = list(self.paths.keys())
        self.selected_keys_chunks: List[List[str]] = []
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
            if len(self.cell_type_specific_samples) == 0:  # может быть др. вид/список
                self.logger.warning(f"No cell-type-specific samples found for {self.targets_path} in {cell_type_specific_samples_path}")
                self.cell_type_specific_samples = None
                self.cell_type_specific_samples_path = None
                self.N_cell_type_specific_samples = 0
            else:
                self.cell_type_specific_samples = self.cell_type_specific_samples.groupby("gene_id")["cell_id"].apply(list).to_dict() # gene_id -> list of cell_ids
                self.n_cell_chunks = 1
        else:
            self.N_cell_type_specific_samples = 0

        self.files_opened = False
        self._bw_key_to_col: Dict[str, int] = {}

        self.sequences = FastaFile(self.genome)

        # путь для токенов
        self.h5_cache_path = self.get_hash_path() + ".h5"

        # кэш токенизации
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
                self.logger.debug(f"Loading tpm cache from {tpm_hash_path}")
                self.tpm_lookup = pickle.load(open(tpm_hash_path, "rb"))
                assert len(self.tpm_lookup) == len(self.paths), "Number of tpm cache and paths are not the same"
                assert all(key in self.tpm_lookup for key in self.paths.keys()), "All keys in paths must be in tpm cache"
                for key, value in self.tpm_lookup.items():
                    assert pd.isna(value).sum().sum()==0, f"TPM cache contains NaN values for {key}"
            else:
                self.tpm_cache = {}
                for key, (bw_paths, tpm_path) in tqdm.tqdm(self.paths.items()):
                    self.logger.debug(f"Reading tpm from {tpm_path}")
                    tpm_df = pd.read_csv(tpm_path, dtype=np.float32)
                    self.tpm_cache[key] = tpm_df
                self.tpm_lookup = {}
                for key, tpm_df in self.tpm_cache.items():
                    self.tpm_lookup[key] = tpm_df.T.set_index(tpm_df.columns)
                pickle.dump(self.tpm_lookup, open(tpm_hash_path, "wb"))

        # валидные индексы
        self.valid_indices: List[int] = []
        self._compute_valid_indices()

    # -------------------- вспомогательные методы --------------------

    def _compute_valid_indices(self):
        """Если TPM отключён — валидны все гены. Иначе — только те, где есть TPM хотя бы в одном ключе."""
        self.logger.debug("Computing valid indices...")
        if not self.tpm:
            self.valid_indices = list(range(len(self.genes)))
            self.logger.info(f"TPM disabled — using all {len(self.valid_indices)} genes as valid")
            return

        for idx in range(len(self.genes)):
            gene_id = self.genes.iloc[idx]['gene_id']
            has_tpm_data = False
            for key in self.paths.keys():
                if gene_id in self.tpm_lookup[key].index:
                    has_tpm_data = True
                    break
            if has_tpm_data:
                self.valid_indices.append(idx)
        self.logger.info(f"Found {len(self.valid_indices)} valid samples out of {len(self.genes)}")

    # def get_hash_path(self):
    #     m = hashlib.blake2b(digest_size=8)
    #     input_strings = []
        
    #     for s in [
    #         'tokens',
    #         str(self.intervals_hash),
    #         str(self.genome),
    #         str(self.num_before),
    #         str(self.token_len_for_fetch),
    #         str(self.gen_max_seq_len),
    #         getattr(self.gen_tokenizer, "name_or_path", "tok")
    #     ]:
    #         m.update(s.encode("utf-8"))
    #         input_strings.append(s)
    #     self.logger.debug(f"Hash inputs: {input_strings}")
    #     self.logger.debug(f"constructed hash suffix: {m.hexdigest()}")
    #     hash_suffix = m.hexdigest()
    #     hash_path = str(self.hash_prefix) + "." + hash_suffix
    #     return hash_path
        
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
        hash_path = str(self.hash_prefix) + "." + hash_suffix
        return hash_path

    def get_signals_hash_path(self):
        m = hashlib.blake2b(digest_size=8)
        m.update(str('signals').encode("utf-8"))
        m.update(str(self.intervals_hash).encode("utf-8"))
        m.update(str(self.targets_path).encode("utf-8"))
        m.update(str(self.genome).encode("utf-8"))
        m.update(str(self.num_before).encode("utf-8"))
        m.update(str(self.gen_max_seq_len).encode("utf-8"))
        if self.norm_bw:
            m.update(str("norm_bw").encode("utf-8"))
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
        self.paths: Dict[str, List[Any]] = {}
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
            
            if self.norm_bw:
                self.logger.info("Reading metadata for normalization of bigwig tracks")
                meta_list = []
                dir_path = os.path.dirname(self.targets_path)
                for _, row in df.iterrows():
                    json_path = os.path.abspath(os.path.join(dir_path, row['metadata']))
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                        meta_list.append(metadata)
                meta_df = pd.DataFrame(meta_list, index=df['id'])
                self.coverage_norm = {
                    k: {
                        "+": meta_df.loc[k]["forward_total_coverage"],
                        "-": meta_df.loc[k]["reverse_total_coverage"]
                    }
                    for k in df['id']
                }
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
        fasta_ref_lengths = {k:v for k, v in zip(sequences.references, sequences.lengths)}
        bw_ref_lengths = {k:v for k, v in bw_handler.chroms().items()}
        common_refs = set(fasta_ref_lengths.keys()) & set(bw_ref_lengths.keys())
        assert len(common_refs) > 0, f"No common references found in genome and bigwig file. Genome: {list(fasta_ref_lengths.keys())}, Bigwig: {list(bw_ref_lengths.keys())}. Genome mismatch?"
        for ref in common_refs:
            if fasta_ref_lengths[ref] != bw_ref_lengths[ref]:
                raise ValueError(f"Length of {ref} in genome and bigwig file are different. Ref: {ref}, Genome: {fasta_ref_lengths[ref]}, Bigwig: {bw_ref_lengths[ref]}")

    def open_files(self):
        """Открытие bigWig и создание быстрого маппинга key->col."""
        if not self.bw or self.files_opened:
            return
        self.bigWigHandlers: Dict[str, Dict[str, Any]] = {}
        for k, (v1, v2) in self.paths.items():
            try:
                self.bigWigHandlers[k] = {strand: bw.open(path) for strand, path in v1.items()}
                for bw_handler in self.bigWigHandlers[k].values():
                    self.check_bw_genome_consistency(bw_handler, self.sequences)
            except Exception:
                self.logger.exception(f"Error opening bigwig file for key={k}, v1={v1}")
        self.files_opened = True
        # быстрый словарь: ключ -> индекс колонки
        self._bw_key_to_col = {k: i for i, k in enumerate(self.bigWigHandlers.keys())}

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
        token_lengths = []
        
        if self.num_before > 0: 
            if (reverse == 0): # forward strand
                try:
                    sequence = self.sequences.fetch(chrom, max(start - self.num_before * self.token_len_for_fetch, 0), start).upper()
                except ValueError as e:
                    self.logger.error(f"Error sequence {i}")
            else: # reverse strand
                chrom_length = self.sequences.get_reference_length(chrom)
                try:
                    sequence = self.sequences.fetch(chrom, start, min(start + self.num_before * self.token_len_for_fetch, chrom_length)).upper()
                    sequence = self.reverse_complement(sequence)
                except ValueError as e:
                    self.logger.error(f"Error sequence {i}")
                
            encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
            encoded_sequence['input_ids'] = encoded_sequence['input_ids'][1:-1]
            encoded_sequence['offset_mapping'] = encoded_sequence['offset_mapping'][1:-1]
            if len(encoded_sequence['input_ids']) < self.num_before:
                self.logger.warning(f"Trying to tokenize seq before TSS, but it's too short: {len(encoded_sequence['input_ids'])} < {self.num_before}; {chrom}: {start}-{end} ({strand})")
            tokens_before = encoded_sequence['input_ids'][-self.num_before:]
            mapping = encoded_sequence['offset_mapping'][-self.num_before:]
            
            for i2, (start_i, end_i) in enumerate(mapping):
                token_id = tokens_before[i2]
                # NB: избегаем магического 5 — используем длину напрямую
                length = end_i - start_i
                token = self.gen_tokenizer.decode([token_id])  
                token_lengths.append((token_id, token, length))
    
        if reverse == 0:
            start_gene = start - sum(t[2] for t in token_lengths)
        else:
            start_gene = end
    
        if reverse == 0:
            try:
                sequence = self.sequences.fetch(chrom, start, end).upper()
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
        else:
            try:
                sequence = self.sequences.fetch(chrom, end, start).upper()
                sequence = self.reverse_complement(sequence)
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
        
        encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        tokens_before = encoded_sequence['input_ids'][1:-1]
        mapping = encoded_sequence['offset_mapping'][1:-1]
       
        for i2, (start_i, end_i) in enumerate(mapping):
            token_id = tokens_before[i2]
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
        return start_gene, token_lengths_df

    def process_region_signals(self, bw_handler, chrom, starts, ends, l, strand):
        reverse = 0 if strand == "+" else 1

        signals = np.zeros(l, dtype=np.float32)
        
        # Границы региона
        if reverse == 0:
            region_start = int(starts[0])
            region_end = int(ends[-1])
        else:
            region_start = int(starts[-1])
            region_end = int(ends[0])
        
        if region_start >= region_end:
            return signals
            
        try:
            intervals = bw_handler.intervals(chrom, region_start, region_end)
            if not intervals:
                return signals
                
            # прямой способ (как в исходнике) — простой и стабильный
            region_size = region_end - region_start
            position_values = np.zeros(region_size, dtype=np.float32)
            for interval_start, interval_end, value in intervals:
                rel_start = max(0, interval_start - region_start)
                rel_end = min(region_size, interval_end - region_start)
                if rel_start < rel_end:
                    position_values[rel_start:rel_end] = value
            
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
                    assert chrom == self.genes.iloc[idx]['chromosome']

                    l = min(len(input_ids), self.gen_max_seq_len)
                    
                    bigwig_signals = np.zeros((l, len(self.bigWigHandlers)), dtype=np.float32)

                    for i_key, (key, bw_pair) in enumerate(self.bigWigHandlers.items()):
                        track_signals = self.process_region_signals(bw_pair[strand], chrom, starts[:l], ends[:l], l, strand)
                    
                        # Нормализация по покрытию
                        if self.norm_bw:
                            try:
                                norm_factor = self.coverage_norm[key][strand]
                                if norm_factor > 0:
                                    track_signals /= norm_factor
                                else:
                                    self.logger.warning(f"Zero normalization factor for {key}, strand {strand}")
                            except KeyError:
                                self.logger.warning(f"Missing normalization factor for {key}, strand {strand}")
                    
                        bigwig_signals[:, i_key] = track_signals

                    
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
        gene_idx = idx // self.n_cell_chunks
        chunk_idx = idx % self.n_cell_chunks

        original_idx = self.valid_indices[gene_idx]
        gene_id = self.genes.iloc[original_idx]['gene_id']
        gene_group = self.h5_cache[gene_id]
        selected_keys = self.selected_keys_chunks[chunk_idx]

        if not self.files_opened and self.bw:
            self.open_files()

        input_ids = np.array(gene_group['input_ids'])
        starts    = np.array(gene_group['starts'])
        ends      = np.array(gene_group['ends'])
        chrom     = gene_group['chrom'][()].decode('utf-8')
        strand    = gene_group.attrs['strand']

        #узнать про genes
        assert strand == self.genes.iloc[original_idx]['strand']
        assert chrom == self.genes.iloc[original_idx]['chromosome']

        l = min(len(input_ids), self.gen_max_seq_len)

        seq_input_ids   = torch.as_tensor(input_ids[:l], dtype=torch.long)
        seq_attn_mask   = torch.ones(l, dtype=torch.long)
        seq_token_types = torch.zeros(l, dtype=torch.long)

        batch_input_ids   = seq_input_ids.unsqueeze(0).repeat(self.n_keys, 1)     # (n_keys, l)
        batch_attn_mask   = seq_attn_mask.unsqueeze(0).repeat(self.n_keys, 1)     # (n_keys, l)
        batch_token_types = seq_token_types.unsqueeze(0).repeat(self.n_keys, 1)   # (n_keys, l)

        if self.bw:
            if self.signals_cache is not None:
                signals_group = self.signals_cache[gene_id]
                bigwig_signals = np.array(signals_group['signals'])  # (l, N_tracks)
            else:
                N_tracks = len(self.bigWigHandlers)
                bigwig_signals = np.zeros((l, N_tracks), dtype=np.float32)
                for i_key, (key, bw_pair) in enumerate(self.bigWigHandlers.items()):
                    track_signals = self.process_region_signals(bw_pair[strand], chrom, starts[:l], ends[:l], l, strand)
                    bigwig_signals[:, i_key] = track_signals

            # индексы нужных колонок
            cols = [self._bw_key_to_col[k] for k in selected_keys]
            # (l, n_keys) -> (n_keys, l)
            labels_np = bigwig_signals[:, cols].T.astype(np.float32, copy=False)
            if self.transform_targets_bw is not None:
                labels_np = self.transform_targets_bw(labels_np)
            labels = torch.from_numpy(labels_np).unsqueeze(-1)   # (n_keys, l, 1)
            labels_mask = torch.ones((self.n_keys, l, 1), dtype=torch.bool)  # (n_keys, l, 1)
        else:
            labels      = torch.zeros((self.n_keys, l, 1), dtype=torch.float32)
            labels_mask = torch.zeros((self.n_keys, l, 1), dtype=torch.bool)

        reverse = 0 if strand == "+" else 1
        if reverse == 0 :
            start_coord = starts[0]
            end_coord   = ends[l-1]
        else:
            start_coord = starts[l-1]
            end_coord   = ends[0]

        # ---- TPM на каждый ключ ----
        tpm_values = np.full(self.n_keys, np.nan, dtype=np.float32)
        if self.tpm:
            for i_key, key in enumerate(selected_keys):
                if gene_id in self.tpm_lookup[key].index:
                    tpm_values[i_key] = float(self.tpm_lookup[key].loc[gene_id].iloc[0])
            if self.transform_targets_tpm is not None:
                tpm_values = self.transform_targets_tpm(tpm_values)

        # среднее/отклонение
        if not np.all(np.isnan(tpm_values)):
            dataset_mean = float(np.nanmean(tpm_values))
            if dataset_mean != 0.0:
                dev = (tpm_values - dataset_mean) / dataset_mean
            else:
                dev = np.zeros_like(tpm_values, dtype=np.float32)
        else:
            dataset_mean = np.nan
            dev = np.full_like(tpm_values, np.nan, dtype=np.float32)

        # desc_vectors на каждый ключ
        desc_vectors_list = []
        for key in selected_keys:
            if key not in self.desc_data:
                raise KeyError(f"Track ID '{key}' not found in desc_data")
            desc_vectors_list.append(self.desc_data[key])

        if isinstance(desc_vectors_list[0], int):
            desc_vectors = np.zeros((self.n_keys, 1), dtype=np.int32)
            for i2, v in enumerate(desc_vectors_list):
                desc_vectors[i2, 0] = v
            desc_t = torch.tensor(desc_vectors, dtype=torch.int32)
        else:
            D = len(desc_vectors_list[0])
            desc_vectors = np.zeros((self.n_keys, D), dtype=np.float32)
            for i2, v in enumerate(desc_vectors_list):
                desc_vectors[i2] = v
            desc_t = torch.tensor(desc_vectors, dtype=torch.float32)

        valid_tpm_count = int(np.count_nonzero(~np.isnan(tpm_values)))

        # собираем «батч по ключам»
        features = {
            # входы батча
            "input_ids": batch_input_ids,          # (n_keys, l) long
            "attention_mask": batch_attn_mask,     # (n_keys, l) long
            "token_type_ids": batch_token_types,   # (n_keys, l) long

            # таргеты батча
            "labels": labels,                      # (n_keys, l) float
            "labels_mask": labels_mask,            # (n_keys, l) bool

            # per-sample на ключ
            "tpm": torch.from_numpy(tpm_values).float().unsqueeze(-1) ,                 # (n_keys,1)
            "dataset_deviation": torch.from_numpy(dev).float(),  # (n_keys,)
            "desc_vectors": desc_t,                              # (n_keys, D or 1)

            # общая мета
            "dataset_mean": torch.tensor(dataset_mean, dtype=torch.float32),  # ()
            "selected_keys": list(selected_keys),      # list[str] длины n_keys
            "gene_id": [gene_id] * self.n_keys,        # список дубликатов (можно хранить один, если удобнее)
            "name": self.genes.iloc[original_idx]['gene_name'],
            "chrom": chrom,
            "reverse": int(reverse),
            "start": int(start_coord),
            "end": int(end_coord),
            "dataset_description": [self.dataset_description] * valid_tpm_count,
            # флаговый тензор сэмпла: базовый датасет — 1
            "dataset_flag": torch.ones(self.n_keys, dtype=torch.float32),
        }

        return features

    def __del__(self):
        try:
            if hasattr(self, 'sequences') and self.sequences is not None:
                self.sequences.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'h5_cache') and self.h5_cache is not None:
                self.h5_cache.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'signals_cache') and self.signals_cache is not None:
                self.signals_cache.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'bigWigHandlers'):
                for d in self.bigWigHandlers.values():
                    for h in d.values():
                        try:
                            h.close()
                        except Exception:
                            pass
        except Exception:
            pass

    # return main info about dataset for logging
    def describe(self):
        result = f"ExpressionDataset(n_genes={len(self.valid_indices)}, n_cell_types={len(self.paths.keys())}, n_chunks={self.n_cell_chunks}, bw={self.bw}, tpm={self.tpm}"
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


class ExpressionDatasetMode2(ExpressionDataset):
    """
    Режим 2: батч состоит из n_keys разных генов (ДНК), таргеты берутся для
    одного и того же cell type (одного ключа из targets_path). Между батчами
    перебираются все cell types и все чанки генов.

    ВАЖНО: в этом режиме параметр n_keys трактуется как "количество генов в батче".
    """

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
        bw: str = "",
        tpm: str = "",
        hash_prefix=None,
        n_keys: Optional[int] = None,
        token_len_for_fetch: int = 8,
        fraction_of_cell_type_specific_tpm_samples: float = 0,
        cell_type_specific_samples_path: str = None,
        norm_bw=False,
    ):
        # Инициализация базового датасета (кэширование токенов/сигналов, чтение путей и т.д.)
        super().__init__(
            gen_tokenizer=gen_tokenizer,
            targets_path=targets_path,
            text_data_path=text_data_path,
            genome=genome,
            forward_intervals_path=forward_intervals_path,
            reverse_intervals_path=reverse_intervals_path,
            loglevel=loglevel,
            seed=seed,
            num_before=num_before,
            gen_max_seq_len=gen_max_seq_len,
            transform_targets_bw=transform_targets_bw,
            transform_targets_tpm=transform_targets_tpm,
            bw=bw,
            tpm=tpm,
            hash_prefix=hash_prefix,
            n_keys=n_keys, 
            token_len_for_fetch=token_len_for_fetch,
            fraction_of_cell_type_specific_tpm_samples=fraction_of_cell_type_specific_tpm_samples,
            cell_type_specific_samples_path=cell_type_specific_samples_path,
            norm_bw=norm_bw,
        )

        # Переопределяем семантику n_keys: теперь это число генов в батче
        self.n_keys = n_keys if n_keys is not None else 8

        # Подготовим индексы валидных генов и разобьём на чанки размера n_keys
        self.all_gene_indices = list(self.valid_indices)
        self.num_gene_chunks = (len(self.all_gene_indices) + self.n_keys - 1) // self.n_keys

        # Все доступные cell_id (ключи)
        self.all_keys = list(self.paths.keys())

    def __len__(self):
        # Для каждого cell_id имеются свои чанки генов
        return len(self.all_keys) * self.num_gene_chunks

    def __getitem__(self, idx):
        # Выбираем cell_id и чанк генов
        cell_chunk = idx // self.num_gene_chunks
        gene_chunk = idx % self.num_gene_chunks
        cell_id = self.all_keys[cell_chunk]

        # Формируем список индексов генов для батча
        start = gene_chunk * self.n_keys
        end = min((gene_chunk + 1) * self.n_keys, len(self.all_gene_indices))
        gene_indices = self.all_gene_indices[start:end]

        # Убедимся, что bigWig открыт при необходимости
        if self.bw and not self.files_opened:
            self.open_files()

        seqs = []
        attns = []
        toktypes = []
        labels_list = []
        masks_list = []
        tpm_vals = []
        names = []
        gene_ids_out = []
        chroms = []
        reverses = []
        starts_meta = []
        ends_meta = []
        lengths = []

        for gi in gene_indices:
            gene_id = self.genes.iloc[gi]['gene_id']
            gene_group = self.h5_cache[gene_id]
            input_ids = np.array(gene_group['input_ids'])
            starts = np.array(gene_group['starts'])
            ends = np.array(gene_group['ends'])
            chrom = gene_group['chrom'][()].decode('utf-8')
            strand = gene_group.attrs['strand']

            assert strand == self.genes.iloc[gi]['strand']
            assert chrom == self.genes.iloc[gi]['chromosome']

            l = min(len(input_ids), self.gen_max_seq_len)
            lengths.append(l)

            # Входы
            seq = torch.as_tensor(input_ids[:l], dtype=torch.long)
            attn = torch.ones(l, dtype=torch.long)
            tokt = torch.zeros(l, dtype=torch.long)
            seqs.append(seq)
            attns.append(attn)
            toktypes.append(tokt)

            # Labels: берём только колонку выбранного cell_id
            if self.bw:
                if hasattr(self, 'signals_cache') and self.signals_cache is not None:
                    sig_mat = np.array(self.signals_cache[gene_id]['signals'])  # (l_full, N_tracks)
                else:
                    N_tracks = len(self.bigWigHandlers)
                    sig_mat = np.zeros((l, N_tracks), dtype=np.float32)
                    for i_key, (k, bw_pair) in enumerate(self.bigWigHandlers.items()):
                        track = self.process_region_signals(bw_pair[strand], chrom, starts[:l], ends[:l], l, strand)
                        sig_mat[:, i_key] = track

                col = self._bw_key_to_col[cell_id]
                lab_np = sig_mat[:l, col].astype(np.float32, copy=False)
                if self.transform_targets_bw is not None:
                    lab_np = self.transform_targets_bw(lab_np[:, None]).squeeze(-1)
                lab = torch.from_numpy(lab_np)
                labels_list.append(lab.unsqueeze(-1))         # (l, 1)
                masks_list.append(torch.ones(l, 1, dtype=torch.bool))  # (l, 1)
            else:
                labels_list.append(torch.zeros(l, 1, dtype=torch.float32))
                masks_list.append(torch.zeros(l, 1, dtype=torch.bool))

            reverse = 0 if strand == "+" else 1
            if reverse == 0:
                start_coord = starts[0]
                end_coord = ends[l - 1]
            else:
                start_coord = starts[l - 1]
                end_coord = ends[0]

            gene_ids_out.append(gene_id)
            names.append(self.genes.iloc[gi]['gene_name'])
            chroms.append(chrom)
            reverses.append(int(reverse))
            starts_meta.append(int(start_coord))
            ends_meta.append(int(end_coord))

            # TPM по выбранному cell_id
            if self.tpm:
                if gene_id in self.tpm_lookup[cell_id].index:
                    tpm_vals.append(float(self.tpm_lookup[cell_id].loc[gene_id].iloc[0]))
                else:
                    tpm_vals.append(np.nan)
            else:
                tpm_vals.append(np.nan)

        # Паддинг по максимальной длине в батче (не больше gen_max_seq_len)
        L = int(min(self.gen_max_seq_len, max(lengths) if len(lengths) > 0 else self.gen_max_seq_len))

        def pad_to_L(t: torch.Tensor, fill: float = 0):
            if t.shape[0] == L:
                return t
            out_shape = (L,) + t.shape[1:]            # поддержка (L, 1) и (L,)
            out = torch.full(out_shape, fill, dtype=t.dtype)
            out[: t.shape[0]] = t
            return out

        pad_id = int(self.gen_tokenizer.pad_token_id) if getattr(self.gen_tokenizer, "pad_token_id", None) is not None else 3
        batch_input_ids = torch.stack([pad_to_L(x, fill=pad_id) for x in seqs], dim=0)
        batch_attention = torch.stack([pad_to_L(x, fill=0) for x in attns], dim=0)
        batch_toktypes = torch.stack([pad_to_L(x, fill=0) for x in toktypes], dim=0)
        batch_labels = torch.stack([pad_to_L(x, fill=0.0) for x in labels_list], dim=0)
        batch_mask = torch.stack([pad_to_L(x, fill=0) for x in masks_list], dim=0)

        # Допаддим по первой оси до n_keys для batch-тензоров
        current_bs = batch_input_ids.shape[0]
        if current_bs < self.n_keys:
            pad_rows = self.n_keys - current_bs
            pad_long_ids = torch.full((pad_rows, L), 8, dtype=torch.long)
            zeros_long   = torch.zeros((pad_rows, L), dtype=torch.long)

            # ВАЖНО: 3D нули для labels и labels_mask
            zeros_float3 = torch.zeros((pad_rows, L, 1), dtype=torch.float32)
            zeros_bool3  = torch.zeros((pad_rows, L, 1), dtype=torch.bool)

            batch_input_ids = torch.cat([batch_input_ids, pad_long_ids], dim=0)
            batch_attention = torch.cat([batch_attention, zeros_long],   dim=0)
            batch_toktypes  = torch.cat([batch_toktypes,  zeros_long],   dim=0)
            batch_labels    = torch.cat([batch_labels,    zeros_float3], dim=0)
            batch_mask      = torch.cat([batch_mask,      zeros_bool3],  dim=0)

        # Сводки по TPM
        # Паддим только TPM до длины n_keys значениями NaN, остальные поля не паддим
        if len(tpm_vals) < self.n_keys:
            tpm_vals.extend([np.nan] * (self.n_keys - len(tpm_vals)))
        tpm_arr = np.asarray(tpm_vals, dtype=np.float32)

        if self.transform_targets_tpm is not None:
            # ожидаем, что transform принимает numpy-массив и возвращает массив той же длины
            tpm_arr = np.asarray(self.transform_targets_tpm(tpm_arr), dtype=np.float32)
        
        if not np.all(np.isnan(tpm_arr)):
            dataset_mean = float(np.nanmean(tpm_arr))
            if dataset_mean != 0.0:
                dev = (tpm_arr - dataset_mean) / dataset_mean
            else:
                dev = np.zeros_like(tpm_arr, dtype=np.float32)
        else:
            dataset_mean = np.nan
            dev = np.full_like(tpm_arr, np.nan, dtype=np.float32)

        # desc-вектор выбранного cell_id, дублируем по фактическому размеру батча
        desc = self.desc_data[cell_id]
        if isinstance(desc, int):
            desc_vectors = torch.tensor(np.tile([[desc]], (self.n_keys, 1)), dtype=torch.int32)
        else:
            desc = np.asarray(desc, dtype=np.float32)
            desc_vectors = torch.tensor(np.tile(desc[None, :], (self.n_keys, 1)), dtype=torch.float32)

        valid_tpm_count = int(np.count_nonzero(~np.isnan(tpm_arr)))

        if len(gene_ids_out) < self.n_keys:
            pad_n = self.n_keys - len(gene_ids_out)
            gene_ids_out += [f"pad_gene_{i}" for i in range(pad_n)]


        features = {
            "input_ids": batch_input_ids.long(),
            "attention_mask": batch_attention.long(),
            "token_type_ids": batch_toktypes.long(),
            "labels": batch_labels.float(),
            "labels_mask": batch_mask.bool(),
            "tpm": torch.from_numpy(tpm_arr).float().unsqueeze(-1) ,
            "dataset_deviation": torch.from_numpy(dev).float(),
            "desc_vectors": desc_vectors,
            "dataset_mean": torch.tensor(dataset_mean, dtype=torch.float32),
            "selected_keys": [cell_id] * len(gene_indices),
            "gene_id": gene_ids_out,
            "name": names,
            "chrom": chroms,
            "reverse": reverses,
            "start": starts_meta,
            "end": ends_meta,
            "dataset_description": [self.dataset_description] * current_bs,
            # флаговый тензор сэмпла: режим 2 — 0
            "dataset_flag": torch.zeros(self.n_keys, dtype=torch.float32),
        }

        return features

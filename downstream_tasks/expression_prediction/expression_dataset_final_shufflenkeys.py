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
import re
import h5py
import tqdm
import sys
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from downstream_tasks.expression_prediction.datasets.src.utils import convert_fm_relative_path_to_absolute_path
from pathlib import Path



class ExpressionDataset(Dataset):
    def __init__(
        self,
        gen_tokenizer,
        targets_path: str,
        genome: str,
        forward_intervals_path: str = None,
        reverse_intervals_path: str = None,
        loglevel: int = logging.WARNING,
        seed: int = 42,
        num_before: int = 512,
        gen_max_seq_len: int = 1024,
        transform_targets_bw=None,
        transform_targets_tpm=None,
        bw : str = "",
        tpm : str = "",
        hash_prefix = None,
        n_keys: Optional[int] = None,
        token_len_for_fetch: int = 10,
        norm_bw = False,
        text_tokenizer: str = "intfloat/multilingual-e5-large-instruct",
        text_max_seq_len: int = 1000
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=loglevel)

        assert sys.version_info >= (3, 8), "Python 3.8+ required"

        if isinstance(gen_tokenizer, str):
            self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_tokenizer, trust_remote_code=True)
        else:
            self.gen_tokenizer = gen_tokenizer

        self.token_len_for_fetch = token_len_for_fetch

        self.gen_max_seq_len = gen_max_seq_len
        self.genome = genome
        self._genome_sizes_map = {}
        genome_sizes_path = Path(
            "/mnt/20tb/aspeedok/GENA_LM/downstream_tasks/expression_prediction/datasets/data/genomes/genome_sizes.tsv"
        )
        if genome_sizes_path.exists():
            df_sizes = pd.read_csv(genome_sizes_path, sep="\t")
            self._genome_sizes_map = dict(zip(df_sizes["path"], df_sizes["size"]))

        self.seed = seed
        np.random.seed(self.seed)
        
        self.bw = bw
        self.tpm = tpm
        self.norm_bw = norm_bw 
        
        self.targets_path = targets_path

        self.num_before = num_before
        self.transform_targets_bw = transform_targets_bw
        self.transform_targets_tpm = transform_targets_tpm

        assert forward_intervals_path is not None or reverse_intervals_path is not None, "Either forward_intervals_path or reverse_intervals_path must be provided"
        self.intervals_hash = self._name_and_size(forward_intervals_path) + "|" + self._name_and_size(reverse_intervals_path)
        if hash_prefix is None:
            self.hash_prefix = os.path.dirname(forward_intervals_path) if forward_intervals_path is not None else os.path.dirname(reverse_intervals_path)
            self.hash_prefix = os.path.join(self.hash_prefix, "dataset_hash")
        else:
            self.hash_prefix = hash_prefix

        self.read_paths()
        self._bw_key_to_col = {k: i for i, k in enumerate(self.paths.keys())}
        
        forward_genes = pd.read_csv(forward_intervals_path, sep=None, engine="python") if forward_intervals_path is not None else pd.DataFrame()
        forward_genes["strand"] = "+"
        reverse_genes = pd.read_csv(reverse_intervals_path, sep=None, engine="python") if reverse_intervals_path is not None else pd.DataFrame()
        reverse_genes["strand"] = "-"
        self.genes = pd.concat([forward_genes, reverse_genes], ignore_index=True)

        if n_keys is None:
            n_keys = len(self.paths.keys())
        self.n_keys = n_keys

        self.n_cell_chunks = ((len(self.paths.keys()) - 1) // n_keys) + 1
        self.all_keys = list(self.paths.keys())
        self.epoch = 0

        self.files_opened = False
        self.sequences = FastaFile(self.genome)
        self.h5_cache_path = self.get_hash_path() + ".h5"

        if os.path.exists(self.h5_cache_path):
            self.h5_cache = h5py.File(self.h5_cache_path, "r")
        else:
            self.precompute_tokenization()


        if self.bw:
            self.signals_cache_path = self.get_signals_hash_path() + ".h5"
            if os.path.exists(self.signals_cache_path):
                self.signals_cache = h5py.File(self.signals_cache_path, "r")
            else:
                self.precompute_signals()
                   
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

        self.valid_indices = []
        if self.bw and not self.tpm: 
            self.valid_indices = list(range(len(self.genes)))   
        else:
            self._compute_valid_indices()
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(
    text_tokenizer,
    padding_side="left"
)
        self.text_max_seq_len = text_max_seq_len
        self.text_data = {}  
        self.text_data_keys = set()
        tokenizer_tag = text_tokenizer.replace("/", "_")
        self.desc_h5_cache_path = f"{os.path.abspath(targets_path)}.{tokenizer_tag}.{text_max_seq_len}.description.h5"

        if os.path.exists(self.desc_h5_cache_path):
            self.desc_h5_cache = h5py.File(self.desc_h5_cache_path, "r")
        else:
            self.load_descriptions_from_json(targets_path)
            self.precompute_descriptions()
            self.desc_h5_cache = h5py.File(self.desc_h5_cache_path, "r")

    def _name_and_size(self, p: str) -> str:
        if p is None:
            return "None|0"
        path = str(p)
        name = Path(path).name
        try:
            size = os.path.getsize(path)  # bytes
        except OSError:
            if path == self.genome:
                genome_size = self._genome_sizes_map.get(path)
                if genome_size is not None:
                    return f"{name}|{genome_size}"
            raise FileNotFoundError(f"File not found for hashing: {path}")
        return f"{name}|{size}"

    def _compute_valid_indices(self):
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
        
    def get_hash_path(self):
        m = hashlib.blake2b(digest_size=8)
        input_strings = []
        
        input_str = str('tokens')
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = str(self.intervals_hash)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = self._name_and_size(self.genome)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = str(self.num_before)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        if self.token_len_for_fetch != 10: # 8 was default in first version of the dataset; TODO: remove at some point
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
        m.update(self._name_and_size(self.targets_path).encode("utf-8"))
        m.update(self._name_and_size(self.genome).encode("utf-8"))
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

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _stable_gene_seed(self, gene_id: str) -> int:
        h = hashlib.blake2b(
            f"{self.seed}|{self.epoch}|{gene_id}".encode("utf-8"),
            digest_size=8
        )
        return int.from_bytes(h.digest(), "little") % (2**32)

    def _get_gene_key_order(self, gene_id: str) -> List[str]:
        rng = np.random.default_rng(self._stable_gene_seed(gene_id))
        perm = rng.permutation(len(self.all_keys))
        return [self.all_keys[i] for i in perm]

    def _get_selected_keys_for_gene(self, gene_id: str, chunk_idx: int) -> List[str]:
        ordered_keys = self._get_gene_key_order(gene_id)
        start_idx = chunk_idx * self.n_keys
        end_idx = min(start_idx + self.n_keys, len(ordered_keys))
        return ordered_keys[start_idx:end_idx]

        
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
                    gene_group.attrs['strand'] = self.genes.iloc[idx]['strand']
                    gene_group.attrs['chrom'] = tokens_df["chrom"].iloc[0]
                    
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
        if getattr(self, "signals_cache", None) is not None:
            self.files_opened = True
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

    def reverse_complement(self, sequence):
        complement = str.maketrans('ACGTN', 'TGCAN')
        return sequence.translate(complement)[::-1]

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
            
        if reverse == 1: 
            token_lengths.reverse()
        token_lengths_df = pd.DataFrame(token_lengths, columns=['token_id', 'token', 'length'])
        token_lengths_df['start'] = token_lengths_df['length'].cumsum().shift(fill_value=0) + start_gene 
        token_lengths_df['end'] = token_lengths_df['start'] + token_lengths_df['length']
        token_lengths_df['chrom'] = chrom
        if reverse == 1: 
            token_lengths_df = token_lengths_df[::-1].reset_index(drop=True)
        return start_gene, token_lengths_df

    # def process_region_signals(self, bw_handler, chrom, starts, ends, l, strand):
    #     reverse = 0 if strand == "+" else 1

    #     signals = np.zeros(l, dtype=np.float32)
        
    #     if reverse == 0:
    #         region_start = int(starts[0])
    #         region_end = int(ends[-1])
    #     else:
    #         region_start = int(starts[-1])
    #         region_end = int(ends[0])
        
    #     if region_start >= region_end:
    #         return signals
            
    #     try:
    #         intervals = bw_handler.intervals(chrom, region_start, region_end)
    #         if not intervals:
    #             return signals
                
    #         region_size = region_end - region_start
    #         position_values = np.zeros(region_size, dtype=np.float32)
    #         for interval_start, interval_end, value in intervals:
    #             rel_start = max(0, interval_start - region_start)
    #             rel_end = min(region_size, interval_end - region_start)
    #             if rel_start < rel_end:
    #                 position_values[rel_start:rel_end] = value
            
    #         for j in range(l):
    #             token_start = max(0, int(starts[j]) - region_start)
    #             token_end = min(region_size, int(ends[j]) - region_start)
    #             if token_start < token_end:
    #                 signals[j] = np.sum(position_values[token_start:token_end])
    #     except Exception as e:
    #         self.logger.error(f"Error processing signals for {chrom}:{region_start}-{region_end}: {e}")
    #     return signals

    def process_region_signals(self, bw_handler, chrom, starts, ends, l, strand):
        reverse = 0 if strand == "+" else 1
        signals = np.zeros(l, dtype=np.float32)

        if reverse == 0:
            region_start, region_end = int(starts[0]), int(ends[-1])
        else:
            region_start, region_end = int(starts[-1]), int(ends[0])

        if region_start >= region_end:
            return signals

        intervals = bw_handler.intervals(chrom, region_start, region_end)
        if not intervals:
            return signals

        region_size = region_end - region_start
        pos = np.zeros(region_size, dtype=np.float32)

        for s, e, v in intervals:
            rs = max(0, s - region_start)
            re = min(region_size, e - region_start)
            if rs < re:
                pos[rs:re] = v

        pref = np.empty(region_size + 1, dtype=np.float32)
        pref[0] = 0.0
        np.cumsum(pos, out=pref[1:])   # pref[i] = sum(pos[:i])

        ts = np.clip(starts[:l].astype(np.int64) - region_start, 0, region_size)
        te = np.clip(ends[:l].astype(np.int64)   - region_start, 0, region_size)
        return pref[te] - pref[ts]

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
                    chrom  = gene_group.attrs['chrom']
                    strand = gene_group.attrs['strand']

                    assert strand == self.genes.iloc[idx]['strand']
                    assert chrom == self.genes.iloc[idx]['chromosome']

                    l = min(len(input_ids), self.gen_max_seq_len)
                    
                    bigwig_signals = np.zeros((l, len(self.bigWigHandlers)), dtype=np.float32)

                    for i_key, (key, bw_pair) in enumerate(self.bigWigHandlers.items()):
                        track_signals = self.process_region_signals(bw_pair[strand], chrom, starts[:l], ends[:l], l, strand)
                    
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

    def load_descriptions_from_json(self, targets_path):
        df = pd.read_csv(targets_path)
        base_dir = os.path.dirname(targets_path)
        for idx, row in df.iterrows():
            description_id = row["id"]
            metadata_path = row["metadata"]
            full_metadata_path = metadata_path if os.path.isabs(metadata_path) else os.path.join(base_dir, metadata_path)
            if not os.path.exists(full_metadata_path):
                raise ValueError(f"Metadata file not found for id '{description_id}': {full_metadata_path}")
            with open(full_metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            desc = ExpressionDataset.make_description_from_json(meta, description_id, full_metadata_path)
            self.text_data[description_id] = desc
        self.text_data_keys = set(self.text_data.keys())

    @staticmethod
    def make_description_from_json(meta, description_id, meta_path):
        if not meta or not isinstance(meta, dict) or len(meta) == 0:
            raise ValueError(f"No description data in metadata for id '{description_id}', file: {meta_path}")
        line_texts = []
        for k, v in meta.items():
            k = k.replace('_', ' ')
            v = str(v).replace('_', ' ')
            clean_k = re.sub(r'^(Characteristics|Chracteristics|Charateristics|Parameter)\\s*', '', k)
            clean_k = re.sub(r'\\[|\\]', '', clean_k).strip()
            clean_k = clean_k if clean_k else k
            clean_v = str(v).replace('"', '').strip()
            line_texts.append(f'{clean_k} is {clean_v}.')
        if not line_texts:
            raise ValueError(f"No description text generated for id '{description_id}', file: {meta_path}")
        return " ".join(line_texts)


    def precompute_descriptions(self):
        print("Precomputing description tokens to {}".format(self.desc_h5_cache_path))
        temp_path = "{}.{}.temp".format(self.desc_h5_cache_path, os.getpid())
        with h5py.File(temp_path, "w") as h5f:
            for description_id, text in tqdm.tqdm(self.text_data.items(), desc="Tokenizing descriptions"):
                encoding = self.text_tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=self.text_max_seq_len,
                    return_tensors="pt"    
                )

                grp = h5f.create_group(str(description_id))
                grp.create_dataset("input_ids", data=encoding["input_ids"][0])
                grp.create_dataset("attention_mask", data=encoding["attention_mask"][0])
            h5f.flush()
        os.rename(temp_path, self.desc_h5_cache_path)

    def __len__(self):
        return len(self.valid_indices) * self.n_cell_chunks

    def __getitem__(self, idx):
        gene_idx = idx // self.n_cell_chunks
        chunk_idx = idx % self.n_cell_chunks

        original_idx = self.valid_indices[gene_idx]
        gene_id = self.genes.iloc[original_idx]['gene_id']
        gene_group = self.h5_cache[gene_id]
        selected_keys = self._get_selected_keys_for_gene(gene_id, chunk_idx)
        n_real = len(selected_keys)

        if self.bw and not self.files_opened:
            self.open_files()

        n_tokens = gene_group["input_ids"].shape[0]
        L = min(n_tokens, self.gen_max_seq_len - 2)
        assert L>0, f"Empty token sequence for gene_id={gene_id}"
        
        input_ids = gene_group["input_ids"][:L]
        starts    = gene_group["starts"][:L]
        ends      = gene_group["ends"][:L]
        chrom  = gene_group.attrs['chrom']
        strand    = gene_group.attrs['strand']
        assert strand == self.genes.iloc[original_idx]['strand']
        assert chrom == self.genes.iloc[original_idx]['chromosome']

        cls_id = self.gen_tokenizer.cls_token_id
        sep_id = self.gen_tokenizer.sep_token_id
        assert (cls_id is not None) and (sep_id is not None), "Tokenizer must have CLS/SEP"

        tok = torch.as_tensor(input_ids, dtype=torch.long)
        seq_input_ids = torch.cat([tok.new_tensor([cls_id]), tok, tok.new_tensor([sep_id])], dim=0)
        seq_attn_mask = torch.ones(seq_input_ids.size(0), dtype=torch.long)
        seq_token_types = torch.zeros(seq_input_ids.size(0), dtype=torch.long)

        batch_input_ids   = seq_input_ids.unsqueeze(0).expand(self.n_keys, -1)
        batch_attn_mask   = seq_attn_mask.unsqueeze(0).expand(self.n_keys, -1)
        batch_token_types = seq_token_types.unsqueeze(0).expand(self.n_keys, -1)

        labels = torch.zeros((self.n_keys, L + 2, 1), dtype=torch.float32)
        labels_mask = torch.zeros((self.n_keys, L + 2, 1), dtype=torch.bool)

        if self.bw:
            if self.signals_cache is not None:
                bigwig_signals = np.array(self.signals_cache[gene_id]['signals'])  
            else:
                N_tracks = len(self.bigWigHandlers)
                bigwig_signals = np.zeros((L, N_tracks), dtype=np.float32)
                for i_key, (key, bw_pair) in enumerate(self.bigWigHandlers.items()):
                    bigwig_signals[:, i_key] = self.process_region_signals(
                        bw_pair[strand], chrom, starts, ends, L, strand
                    )
            cols = [self._bw_key_to_col[k] for k in selected_keys]
            bw_np = bigwig_signals[:L, cols].T  # (n_keys, L)
            if self.transform_targets_bw is not None:
                bw_np = self.transform_targets_bw(bw_np)
            labels[:n_real, 1:1+L, 0] = torch.from_numpy(bw_np)
            labels_mask[:n_real, 1:1+L, 0] = True

        tpm_values = np.full(self.n_keys, np.nan, dtype=np.float32)
        if self.tpm:
            for i_key, key in enumerate(selected_keys):
                if gene_id in self.tpm_lookup[key].index:
                    tpm_values[i_key] = float(self.tpm_lookup[key].loc[gene_id].iloc[0])
            if self.transform_targets_tpm is not None:
                tpm_values = self.transform_targets_tpm(tpm_values)

        tpm_mask = ~np.isnan(tpm_values)                                   # (n_keys,)
        tpm_filled = np.where(tpm_mask, tpm_values, 0.0).astype(np.float32)

        filtered_keys = [k for k, m in zip(selected_keys, tpm_mask) if m]

        labels[:, 0, 0] = torch.from_numpy(tpm_filled)
        labels_mask[:, 0, 0] = torch.from_numpy(tpm_mask).bool()

        reverse = 0 if strand == "+" else 1
        if reverse == 0 :
            start_coord = starts[0]
            end_coord   = ends[L-1]
        else:
            start_coord = starts[L-1]
            end_coord   = ends[0]

        valid_tpm_count = int(np.count_nonzero(~np.isnan(tpm_values)))

        desc_input_ids = []
        desc_attention_mask = []
        for key in selected_keys:
            grp = self.desc_h5_cache[str(key)]
            ids = torch.tensor(grp["input_ids"][()], dtype=torch.long)      
            mask = torch.tensor(grp["attention_mask"][()], dtype=torch.long) 
            desc_input_ids.append(ids)
            desc_attention_mask.append(mask)

        pad_text = "this is just padding"
        pad_enc = self.text_tokenizer(
            pad_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.text_max_seq_len,
            padding=False,
            return_tensors="pt"
        )

        self._pad_desc_ids  = pad_enc["input_ids"][0].to(torch.long)         # (Dp,)
        self._pad_desc_mask = pad_enc["attention_mask"][0].to(torch.long)    # (Dp,)

        n_missing = self.n_keys - len(desc_input_ids)
        for _ in range(n_missing):
            desc_input_ids.append(self._pad_desc_ids.clone())
            desc_attention_mask.append(self._pad_desc_mask.clone())


        # n_missing = self.n_keys - len(desc_input_ids)
        # for _ in range(n_missing):
        #     desc_input_ids.append(torch.tensor([20], dtype=torch.long))  
        #     desc_attention_mask.append(torch.tensor([1], dtype=torch.long))

        if self.bw and not self.tpm: 
            features_selected_keys = []
        else:
            features_selected_keys = filtered_keys

        features = {
            "input_ids": batch_input_ids,          
            "attention_mask": batch_attn_mask,    
            "token_type_ids": batch_token_types,  
            "labels": labels,                    
            "labels_mask": labels_mask,                    
            "selected_keys": filtered_keys,      
            "gene_id": [gene_id] * len(filtered_keys),       
            "name": self.genes.iloc[original_idx]['gene_name'],
            "chrom": chrom,
            "reverse": reverse,
            "start": start_coord,
            "end": end_coord,
            "dataset_description": [self.dataset_description] * self.n_keys,
            "dataset_flag": torch.ones(self.n_keys, dtype=torch.float32),
            "desc_input_ids": desc_input_ids,         
            "desc_attention_mask": desc_attention_mask
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

    def describe(self):
        result = f"ExpressionDataset(n_genes={len(self.valid_indices)}, n_cell_types={len(self.paths.keys())}, n_chunks={self.n_cell_chunks}, bw={self.bw}, tpm={self.tpm}"
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
    def __init__(
        self,
        gen_tokenizer,
        targets_path: str,
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
        token_len_for_fetch: int = 10,
        norm_bw=False,
        text_tokenizer: str = "intfloat/multilingual-e5-large-instruct",
        text_max_seq_len: int = 1000
    ):

        super().__init__(
            gen_tokenizer=gen_tokenizer,
            targets_path=targets_path,
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
            norm_bw=norm_bw,
            text_tokenizer = text_tokenizer,
            text_max_seq_len = text_max_seq_len
        )
        if n_keys is None:
            df_targets = pd.read_csv(targets_path)
            n_keys = len(df_targets)
        self.n_keys = n_keys
        self.num_gene_chunks = (len(self.valid_indices) + self.n_keys - 1) // self.n_keys
        self.all_keys = list(self.paths.keys())
        pad_id = getattr(self.gen_tokenizer, "pad_token_id", None)
        self._pad_id = 0 if pad_id is None else pad_id
        self._pad_desc_ids = torch.tensor([20], dtype=torch.long)
        self._pad_desc_mask = torch.tensor([1], dtype=torch.long)

    @staticmethod
    def _pad_to_len(t: torch.Tensor, L: int, fill):
        if t.shape[0] == L:
            return t
        out = torch.full((L,) + t.shape[1:], fill, dtype=t.dtype)
        out[: t.shape[0]] = t
        return out

    def __len__(self):
        return len(self.all_keys) * self.num_gene_chunks

    def __getitem__(self, idx):
        cell_chunk = idx // self.num_gene_chunks
        gene_chunk = idx % self.num_gene_chunks
        cell_id = self.all_keys[cell_chunk]

        start_idx = gene_chunk * self.n_keys
        end_idx = min((gene_chunk + 1) * self.n_keys, len(self.valid_indices))
        gene_indices = self.valid_indices[start_idx:end_idx]
        n_real = len(gene_indices)
        assert n_real > 0, "Empty gene_indices chunk"

        if self.bw and not self.files_opened:
            self.open_files()

        cls_id = self.gen_tokenizer.cls_token_id or self.gen_tokenizer.bos_token_id
        sep_id = self.gen_tokenizer.sep_token_id or self.gen_tokenizer.eos_token_id
        assert (cls_id is not None) and (sep_id is not None), "Tokenizer must have CLS/SEP"

        col = self._bw_key_to_col[cell_id] if (self.bw and self._bw_key_to_col) else None

        seqs, attns, toktypes = [], [], []
        labels_list, masks_list = [], []
        tpm_list = []

        names, gene_ids, chroms = [], [], []
        reverses, starts_meta, ends_meta = [], [], []
        lengths = []

        for gi in gene_indices:
            row = self.genes.iloc[gi]
            gene_id = row["gene_id"]
            gene_group = self.h5_cache[gene_id]

            n_tokens = gene_group["input_ids"].shape[0]
            L = min(n_tokens, self.gen_max_seq_len - 2)
            assert L > 0, f"Empty token sequence for gene_id={gene_id}"

            input_ids = gene_group["input_ids"][:L]
            starts    = gene_group["starts"][:L]
            ends      = gene_group["ends"][:L]
            chrom  = gene_group.attrs['chrom']
            strand = gene_group.attrs["strand"]
            assert strand == row["strand"]
            assert chrom  == row["chromosome"]

            tok = torch.as_tensor(input_ids, dtype=torch.long)
            seq = torch.cat([tok.new_tensor([cls_id]), tok, tok.new_tensor([sep_id])], dim=0)  # (L+2,)
            attn = torch.ones(seq.size(0), dtype=torch.long)
            tokt = torch.zeros(seq.size(0), dtype=torch.long)

            seqs.append(seq)
            attns.append(attn)
            toktypes.append(tokt)
            lengths.append(seq.size(0))

            lab2 = torch.zeros(seq.size(0), 1, dtype=torch.float32)
            msk2 = torch.zeros(seq.size(0), 1, dtype=torch.bool)

            if self.bw:
                assert col is not None
                if getattr(self, "signals_cache", None) is not None:
                    lab_np = self.signals_cache[gene_id]["signals"][:L, col].astype(np.float32, copy=False)  # (L,)
                else:
                    bw_handler = self.bigWigHandlers[cell_id][strand]
                    lab_np = self.process_region_signals(bw_handler, chrom, starts, ends, L, strand)  # (L,)

                if self.transform_targets_bw is not None:
                    lab_np = self.transform_targets_bw(lab_np[:, None]).squeeze(-1)

                lab2[1:1+L, 0] = torch.from_numpy(lab_np)
                msk2[1:1+L, 0] = True

            labels_list.append(lab2)
            masks_list.append(msk2)

            reverse = 0 if strand == "+" else 1
            if reverse == 0:
                start_coord = int(starts[0])
                end_coord   = int(ends[L-1])
            else:
                start_coord = int(starts[L-1])
                end_coord   = int(ends[0])

            gene_ids.append(gene_id)
            names.append(row["gene_name"])
            chroms.append(chrom)
            reverses.append(int(reverse))
            starts_meta.append(start_coord)
            ends_meta.append(end_coord)

            if self.tpm:
                try:
                    tpm_list.append(float(self.tpm_lookup[cell_id].loc[gene_id].iloc[0]))
                except KeyError:
                    tpm_list.append(np.nan)
            else:
                tpm_list.append(np.nan)

        Lmax = max(lengths) if lengths else 2

        batch_input_ids   = torch.stack([self._pad_to_len(x, Lmax, self._pad_id) for x in seqs], dim=0)
        batch_attn_mask   = torch.stack([self._pad_to_len(x, Lmax, 0) for x in attns], dim=0)
        batch_token_types = torch.stack([self._pad_to_len(x, Lmax, 0) for x in toktypes], dim=0)
        batch_labels      = torch.stack([self._pad_to_len(x, Lmax, 0.0) for x in labels_list], dim=0)
        batch_mask        = torch.stack([self._pad_to_len(x, Lmax, False) for x in masks_list], dim=0)

        current_bs = batch_input_ids.shape[0]
        if current_bs < self.n_keys:
            pad_rows = self.n_keys - current_bs
            batch_input_ids   = torch.cat([batch_input_ids, torch.full((pad_rows, Lmax), self._pad_id, dtype=torch.long)], dim=0)
            batch_attn_mask   = torch.cat([batch_attn_mask, torch.zeros((pad_rows, Lmax), dtype=torch.long)], dim=0)
            batch_token_types = torch.cat([batch_token_types, torch.zeros((pad_rows, Lmax), dtype=torch.long)], dim=0)
            batch_labels      = torch.cat([batch_labels, torch.zeros((pad_rows, Lmax, 1), dtype=torch.float32)], dim=0)
            batch_mask        = torch.cat([batch_mask, torch.zeros((pad_rows, Lmax, 1), dtype=torch.bool)], dim=0)

            tpm_list.extend([np.nan] * pad_rows)

        tpm_values = np.asarray(tpm_list, dtype=np.float32)
        if self.transform_targets_tpm is not None:
            tpm_values = np.asarray(self.transform_targets_tpm(tpm_values), dtype=np.float32)

        tpm_t  = torch.from_numpy(tpm_values)    
        tpm_ok = ~torch.isnan(tpm_t)                       
        batch_labels[:, 0, 0] = torch.where(tpm_ok, tpm_t, torch.zeros_like(tpm_t))
        batch_mask[:, 0, 0]   = tpm_ok

        grp = self.desc_h5_cache[str(cell_id)]
        desc_ids = torch.tensor(grp["input_ids"][()], dtype=torch.long)
        desc_msk = torch.tensor(grp["attention_mask"][()], dtype=torch.long)
        desc_input_ids = [desc_ids.clone() for _ in range(self.n_keys)]
        desc_attention_mask = [desc_msk.clone() for _ in range(self.n_keys)]

        if self.bw and not self.tpm: 
            features_gene_id = []
        else:
            features_gene_id = gene_ids

        features = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attn_mask,
            "token_type_ids": batch_token_types,
            "labels": batch_labels,
            "labels_mask": batch_mask,
            "selected_keys": [cell_id] * len(features_gene_id),
            "gene_id": features_gene_id,
            "name": names,
            "chrom": chroms,
            "reverse": reverses,
            "start": starts_meta,
            "end": ends_meta,
            "dataset_description": [self.dataset_description] * self.n_keys,
            "dataset_flag": torch.zeros(self.n_keys, dtype=torch.float32),  
            "desc_input_ids": desc_input_ids,
            "desc_attention_mask": desc_attention_mask,
        }
        return features

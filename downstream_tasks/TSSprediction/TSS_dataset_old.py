import os
import hashlib
import logging
import numpy as np
import pandas as pd
import h5py
import tqdm
import torch
from torch.utils.data import Dataset
from pysam import FastaFile
from pathlib import Path
import threading
from transformers import AutoTokenizer

import sys
from typing import List, Dict


from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
		"""
		Collate function for DataLoader.
		
		Args:
			batch: List of samples from __getitem__
			
		Returns:
			Batched data
		"""
		batched = {
			'input_ids': torch.stack([item['input_ids'] for item in batch]),
			'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
			'targets': {},
		}
		
		# Batch targets
		if batch[0]['targets'] is None:
			assert np.all([item['targets'] is None for item in batch]), "All targets must be None for inference"
			batched['targets'] = None
		else:
			target_keys = batch[0]['targets'].keys()
			for key in target_keys:
				batched['targets'][key] = torch.stack([item['targets'][key] for item in batch])

		return batched	


class TSSDataset(Dataset):
    """
    Dataset for TSS (transcription start site) prediction.

    Precomputes tokenization of genomic regions around TSS and stores results in HDF5.
    Each sample yields a fixed-length sequence (1024 tokens) with CLS and SEP tokens,
    and a binary label for the TSS position (first token after CLS).
    """
    def __init__(
        self,
        mapping_file: str,
        tokenizer,
        num_before: int = 512,
        token_len_for_fetch: int = 10,
        max_seq_len: int = 1024,
        cache_dir: str|None = None,
        loglevel: int = logging.WARNING,
        data_workers: int = 10,
    ):
        """
        Args:
            mapping_file: CSV file with columns:
                genome_path, chromosome, TSS, taxon, is_TSS, strand
            tokenizer: DNA tokenizer (AutoTokenizer)
            num_before: number of tokens upstream of TSS (rest downstream)
            token_len_for_fetch: approximate length (bp) per token
            max_seq_len: total sequence length including CLS/SEP (must be >= 2)
            cache_dir: directory to store HDF5 cache (default: same as mapping_file)
            loglevel: logging level
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(loglevel)

        # Basic settings
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        self.num_before = num_before
        self.token_len_for_fetch = token_len_for_fetch
        self.max_seq_len = max_seq_len

        # Number of non‑special tokens (excluding CLS and SEP)
        self.num_tokens_no_special = max_seq_len - 2
        self.num_downstream = self.num_tokens_no_special - num_before
        if self.num_downstream < 0:
            raise ValueError(f"num_before ({num_before}) must be <= {self.num_tokens_no_special}")

        # Read mapping file
        self.df = pd.read_csv(mapping_file)
        required_cols = {'genome_path', 'chromosome', 'TSS', 'taxon', 'is_TSS', 'strand'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Mapping file must contain columns: {required_cols}")
        self.num_samples = len(self.df)

        # Prepare cache path
        if cache_dir is None:
            cache_dir = os.path.dirname(mapping_file)
        self.cache_dir = cache_dir
        self.hash_prefix = os.path.join(cache_dir, "tss_dataset")
        self.cache_path = self.get_hash_path() + ".h5"
        
        
        self.data_workers = data_workers
        # Special token IDs
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        
        
        # Open or create cache
        if os.path.exists(self.cache_path):
            self.logger.info(f"Loading cache from {self.cache_path}")
            self.h5_cache = h5py.File(self.cache_path, 'r')
        else:
            self.logger.info(f"Precomputing tokenization to {self.cache_path}")
            self.precompute_tokenization()


        if self.cls_id is None or self.sep_id is None:
            raise ValueError("Tokenizer must have CLS/SEP")
        
        
        

    def _name_and_size(self, path: str) -> str:
        """Return 'filename|size' for a file, used in hash."""
        if path is None:
            return "None|0"
        name = Path(path).name
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        return f"{name}|{size}"

    def get_hash_path(self) -> str:
        """Generate a hash for the cache based on all relevant parameters."""
        m = hashlib.blake2b(digest_size=8)
        m.update(b"tss_dataset_v1")
        m.update(self._name_and_size(self.cache_dir).encode())
        m.update(str(self.num_before).encode())
        m.update(str(self.token_len_for_fetch).encode())
        m.update(str(self.max_seq_len).encode())

        # Include tokenizer name and the whole mapping file content hash
        if hasattr(self.tokenizer, "name_or_path"):
            m.update(self.tokenizer.name_or_path.encode())
        m.update(str(self.num_samples).encode())
        # To be safe, include the mapping file's name and size
        m.update(self._name_and_size(self.cache_dir + "/mapping.csv").encode())

        # Include the content of the mapping file (checksum of entire file)
        try:
            with open(self.cache_dir + "/mapping.csv", "rb") as f:
                m.update(f.read())
        except Exception:
            pass

        hash_suffix = m.hexdigest()
        return str(self.hash_prefix) + "." + hash_suffix

    def reverse_complement(self, seq: str) -> str:
        """Return the reverse complement of a DNA string."""
        comp = str.maketrans('ACGTN', 'TGCAN')
        return seq.translate(comp)[::-1]

    def tokenize_sequence(self, row: pd.Series) -> list:
        """
        Tokenize the genomic region around the TSS for one sample.

        Returns:
            List of token IDs (int). Length may be less than num_tokens_no_special
            if the region is too short; it will be padded later.
        """
        genome_path = row['genome_path']
        chrom = row['chromosome']
        tss = int(row['TSS'])
        strand = row['strand']
        reverse = (strand == '-')

        # Open the genome file for this row
        with FastaFile(genome_path) as genome:
            # Upstream region (relative to transcription direction)
            up_bp = self.num_before * self.token_len_for_fetch
            down_bp = self.num_downstream * self.token_len_for_fetch

            if not reverse:
                # Forward strand: upstream = left, downstream = right
                up_start = max(tss - up_bp, 0)
                up_end = tss
                down_start = tss
                down_end = min(tss + down_bp, genome.get_reference_length(chrom))

                # Fetch upstream sequence and tokenize
                up_seq = genome.fetch(chrom, up_start, up_end).upper()
                up_tokens = self.tokenizer.encode_plus(up_seq)['input_ids'][1:-1]

                # Downstream sequence
                down_seq = genome.fetch(chrom, down_start, down_end).upper()
                down_tokens = self.tokenizer.encode_plus(down_seq)['input_ids'][1:-1]

                # Combine: upstream tokens (last num_before) + downstream tokens
                # If upstream has more than num_before, take only the last num_before
                if len(up_tokens) > self.num_before:
                    up_tokens = up_tokens[-self.num_before:]
                token_ids = up_tokens + down_tokens

            else:
                # Reverse strand: transcription goes leftwards on the positive strand
                # Upstream of TSS on negative strand = further to the right
                up_start = tss
                up_end = min(tss + up_bp, genome.get_reference_length(chrom))
                # Downstream of TSS = further to the left (lower coordinates)
                down_start = max(tss - down_bp, 0)
                down_end = tss

                # Upstream region (right side) – this becomes the first tokens
                up_seq = genome.fetch(chrom, up_start, up_end).upper()
                up_seq = self.reverse_complement(up_seq)
                up_tokens = self.tokenizer.encode_plus(up_seq)['input_ids'][1:-1]

                # Downstream region (left side) – this becomes the later tokens
                down_seq = genome.fetch(chrom, down_start, down_end).upper()
                down_seq = self.reverse_complement(down_seq)
                down_tokens = self.tokenizer.encode_plus(down_seq)['input_ids'][1:-1]

                # Upstream (right side) first, then downstream (left side)
                # For reverse strand, the TSS is at the left end of the region?
                # The original code reversed the whole token list. To keep the same
                # order as forward strand (TSS near the beginning), we put the upstream
                # tokens (which are to the right of TSS) first, then downstream.
                if len(up_tokens) > self.num_before:
                    up_tokens = up_tokens[-self.num_before:]
                token_ids = up_tokens + down_tokens
        
        #print('token ids:', token_ids)
        #print('len of token ids:',len(token_ids))
        #print('number of uptokens',len(up_tokens))
        #print('number of down tokens:',len(down_tokens))
        #print('cls token id',self.cls_id)
        #print('sep token id',self.sep_id)
        #
        #sys.exit(0)
        

        # The tokenizer may have produced more or fewer tokens than needed.
        # We will truncate later to exactly num_tokens_no_special (or pad).
        return token_ids

    def precompute_tokenization(self):
        """Precompute all token sequences and labels, store in HDF5."""
        temp_path = f"{self.cache_path}.{os.getpid()}.temp"

        with h5py.File(temp_path, "w") as h5f:
            pbar = tqdm.tqdm(total=self.num_samples, desc="Tokenizing")
            for idx, row in self.df.iterrows():
                # Tokenize
                token_ids = self.tokenize_sequence(row)

                # Group name is the index (0..N-1)
                grp = h5f.create_group(str(idx))
                # Store token IDs
                grp.create_dataset('token_ids', data=np.array(token_ids, dtype=np.int32))
                # Store label (is_TSS)
                grp.create_dataset('label', data=np.array(row['is_TSS'], dtype=np.int8))
                # Store taxon as attribute
                grp.attrs['taxon'] = row['taxon']
                # Also store genome path? Not needed, but we can add for debugging
                # grp.attrs['genome_path'] = row['genome_path']

                pbar.update(1)
                if idx % 100 == 0:
                    h5f.flush()
            pbar.close()
            h5f.flush()

        # Atomic rename
        os.rename(temp_path, self.cache_path)
        self.h5_cache = h5py.File(self.cache_path, 'r')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Returns a tuple:
            input_ids: torch.LongTensor of shape (max_seq_len,)
            attention_mask: torch.LongTensor of shape (max_seq_len,)
            labels: torch.FloatTensor of shape (max_seq_len, 1)
            labels_mask: torch.BoolTensor of shape (max_seq_len, 1)
            taxon: str (or int if you convert)
        """
        # Load from cache
        grp = self.h5_cache[str(idx)]
        token_ids = grp['token_ids'][()]  # numpy array of ints
        label = int(grp['label'][()])
        taxon = grp.attrs['taxon']

        # Truncate to num_tokens_no_special
        token_ids = token_ids[:self.num_tokens_no_special]
        L = len(token_ids)

        # Add CLS and SEP
        seq_ids = [self.cls_id] + token_ids.tolist() + [self.sep_id]
        seq_len = len(seq_ids)

        # Pad to max_seq_len
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            seq_ids += [self.pad_id] * pad_len
            attn_mask = [1] * seq_len + [0] * pad_len
        else:
            attn_mask = [1] * self.max_seq_len

        input_ids = torch.tensor(seq_ids, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask, dtype=torch.long)

        # Label: only the first token (CLS) should be predicted
        # We set it at position 0 (CLS token)
        labels = torch.zeros(self.max_seq_len, 1, dtype=torch.float32)
        labels_mask = torch.zeros(self.max_seq_len, 1, dtype=torch.bool)

        # The label goes to the CLS token (index 0)
        labels[0, 0] = float(label)
        labels_mask[0, 0] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "labels_mask": labels_mask,
            "taxon": taxon,
        }

    def __del__(self):
        try:
            if hasattr(self, 'h5_cache') and self.h5_cache is not None:
                self.h5_cache.close()
        except Exception:
            pass
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
from transformers import AutoTokenizer

import polars as pl

import random

from typing import List, Dict

#'input_ids': input_ids,
#'attention_mask': attention_mask,
#'labels': labels,
#'labels_mask': labels_mask,
#'taxon': taxon,



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
            'taxon': torch.stack([torch.tensor(item['taxon']) for item in batch]),
            'labels_mask': torch.stack([item['labels_mask'] for item in batch]),
			'labels': torch.stack([item['labels'] for item in batch]),
		}

		return batched	



class CreDataset(Dataset):
    def __init__(self, 
                genome:str,
                chrom_file:str,
                tokenizer: str, 
                token_len_for_fetch:int, 
                max_seq_len:int, 
                cache_dir:str, 
                inputType:str,
                taxon:str,
                loglevel:int= logging.WARNING,
                unknown_taxon_prob:float = 0.15,
                min_cre_number:int|None = None, 
                min_cre_coverage:float|None = None,
                annotation:str|None = None,
                cre_bed:str|None = None
                ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        self.annotation = annotation
        self.genome = genome
        self.taxon = self.encode_taxon(taxon)
        self.cre_bed_unchanged = pl.read_csv(cre_bed, separator='\t', has_header=False, new_columns=('chrom', 'start', 'end', 'idx1', 'idx2', 'type')) \
                            .select('chrom', 'start', 'end') \
                            .sort(by=('chrom', 'start', 'end')) if cre_bed is not None else None
        self.cre_bed = self.merge_intervals(self.cre_bed_unchanged) if cre_bed is not None else None
        self.min_cre_number = min_cre_number
        self.min_cre_coverage = min_cre_coverage
        self.token_len_for_fetch = token_len_for_fetch
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        self.unknown_taxon_prob = unknown_taxon_prob
        
        self.num_special_tokens = 2
        self.interval_len_tokens = self.max_seq_len - self.num_special_tokens
        
        self.hash_prefix = os.path.join(cache_dir, f'{inputType}_'+os.path.basename(chrom_file).split('.')[0])
        
        self.chrom2TSSlist = None
        self.inputType = inputType
        
        self.h5_cache_path = self.get_hash_path() + '.h5'
        
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        
        with open(chrom_file) as handle:
            self.chroms = [chrom.removesuffix('\n') for chrom in handle.readlines()]
        
        
        if os.path.exists(self.h5_cache_path):
            self.h5_cache = h5py.File(self.h5_cache_path, 'r')
        else:
            self.precompute_intervals_tokinization()
            

        if self.inputType == 'TSS':
            assert (self.genome is not None) & (self.annotation is not None)

            
        elif self.inputType == 'CRE':
            assert (self.genome is not None) & (self.cre_bed is not None) & (self.min_cre_number is not None)
            
    def __len__(self):
        return len(self.h5_cache)
            

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
        m.update(b"dataset")
        m.update(self._name_and_size(self.cache_dir).encode())
        m.update(str(self.token_len_for_fetch).encode())
        m.update(str(self.max_seq_len).encode())

        # Include tokenizer name and the whole mapping file content hash
        if hasattr(self.tokenizer, "name_or_path"):
            m.update(self.tokenizer.name_or_path.encode())
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
    
    def encode_taxon(self, taxon) -> int: #TODO
        mapping = {'Unknown': 0,
                    "Lepidosauria": 1,
                    "Chondrichthyes": 2,
                    "Mammalia": 3,
                    "Amphibia": 4,
                    "Actinopteri": 5,
                    "Myxini": 6,
                    "Aves": 7}
        
        return mapping[taxon]

    def merge_intervals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Merge overlapping or adjacent intervals per chromosome.
        Input df must have columns: chrom, start, end
        """
        return (
            df.sort(["chrom", "start"])
            .with_columns(
                group_id = (
                    (pl.col("start") > pl.col("end").cum_max().shift(1).fill_null(0)) 
                    | (pl.col("chrom") != pl.col("chrom").shift(1).fill_null(""))
                ).cum_sum()
            )
            .group_by(["chrom", "group_id"])
            .agg(
                pl.col("start").min().alias("start"),
                pl.col("end").max().alias("end")
            )
            .select("chrom", "start", "end")
            .sort(["chrom", "start"])
        )

    def compute_coverage_merged(self, chrom: str, start: int, end: int) -> float:
        """
        Compute coverage fraction for a region using pre-merged intervals.
        df_merged must have columns: chrom, start, end (non-overlapping)
        """
        region_len = end - start
        if region_len <= 0:
            return 0.0

        subset = (
            self.cre_bed.filter(
                (pl.col("chrom") == chrom) & 
                (pl.col("start") < end) & 
                (pl.col("end") > start)
            )
            .select(
                (pl.col("start").clip(start, end)).alias("s"),
                (pl.col("end").clip(start, end)).alias("e")
            )
        )

        if subset.is_empty():
            return 0.0

        covered_len = (subset["e"] - subset["s"]).sum()
        return float(covered_len) / region_len
    
    def extract_TSS(self):
        TSSbed=[]

        f = open(self.annotation)
        for line in f:
            table = line.split('\t')
            if len(table) < 3:
                continue
            if table[2] == 'transcript':
                chrom  = table[0]
                strand = table[6] 
                geneid = line.split('gene_id')[1].split('"')[1]
                if "gene_name" in line:
                    genesymbol = line.split('gene_name')[1].split('"')[1]
                else:
                    genesymbol = geneid
                
                if strand == "+":
                    iregion = {'chrom':chrom, 'TSS':int(table[3])-1, 'strand':strand, 'geneid':geneid, 'gene_symbol':genesymbol, 'annotation': os.path.relpath(self.annotation)}
                elif strand == '-':
                    iregion = {'chrom':chrom, 'TSS':int(table[4]), 'strand':strand, 'geneid':geneid, 'gene_symbol':genesymbol, 'annotation': os.path.relpath(self.annotation)}
                TSSbed.append(iregion)
        
        f.close()
        return pl.DataFrame(TSSbed).unique()
    
    def make_chrom2TSSlist(self):
        
        if self.chrom2TSSlist is None:
            tss = self.extract_TSS()
            chrom2TSSlist = {}
            for chrom, tss_list in tss.group_by('chrom').agg('TSS').with_columns(TSS = pl.col('TSS').list.sort()).iter_rows():
                chrom2TSSlist[chrom] = tss_list
            self.chrom2TSSlist = chrom2TSSlist
        
        return self.chrom2TSSlist
    
    
    def check_if_contains_TSS(self, chrom, start, end) -> int:
        
        try:
            TSSlist = self.make_chrom2TSSlist()[chrom]
        except:
            return 0
        for TSS in TSSlist:
            if start < TSS < end:
                return 1
        return 0
    
    def check_if_cre_enriched(self, chrom, start, end):
        count = self.cre_bed.filter(pl.col('chrom') == chrom).with_columns(isIntersecting = (pl.col('start') >= start) & pl.col('end') <= end)['isIntersecting'].sum()
        return (count >= self.min_cre_number) 
    
    def check_if_exceeds_min_coverage(self, chrom, start, end):
        return (self.compute_coverage_merged(chrom, start, end) > self.min_cre_coverage)
        
   
    def compute_encoded_seq_len(self, tokens, mapping):
        
        tokenized_sequence_len = 0
        for i, (start, end) in enumerate(mapping):
            token_id = tokens[i]
            if token_id == 5:
                if i > 0:
                    token_len = end - mapping[i-1][1]
                else:
                    token_len = end
            else:
                token_len = end - start
                
            tokenized_sequence_len += token_len
        return tokenized_sequence_len
    
    def precompute_intervals_tokinization(self):
        
        """
        1) iterate over chroms
        2) tokenize sequence into 1022 tokens
        3) get end of tokenized sequence
        4) tokenize new sequence (1022 tokens) starting from end of the previous one
        """
        
        temp_path = f"{self.h5_cache_path}.{os.getpid()}.tmp"
        
        try:
            with h5py.File(temp_path, 'w') as h5f:
                #pbar = tqdm.tqdm(desc='Tokenizing intervals')
                with FastaFile(self.genome) as genome:
                    
                    idx = 0
                    for chrom in self.chroms:
                        chrom_len = genome.get_reference_length(chrom)
                        start = 0
                        while start < chrom_len:
                            end = min(start+self.interval_len_tokens*self.token_len_for_fetch, chrom_len)
                            sequence = genome.fetch(chrom, start, end).upper()
                            
                            encoded_sequence = self.tokenizer.encode_plus(sequence, return_offsets_mapping=True)
                            encoded_sequence['input_ids'] = encoded_sequence['input_ids'][1:-1]
                            encoded_sequence['offset_mapping'] = encoded_sequence['offset_mapping'][1:-1]
                            
                            if encoded_sequence['input_ids'].__len__() < self.interval_len_tokens:
                                #self.logger.warning(f"len of token sequence {encoded_sequence['input_ids'].__len__()} is less then {self.interval_len_tokens}")
                                pass
                                
                            mapping = encoded_sequence['offset_mapping'][0:self.interval_len_tokens]
                            tokens = encoded_sequence['input_ids'][0:self.interval_len_tokens]
                            
                            tokenized_sequence_len = self.compute_encoded_seq_len(tokens, mapping)
                            
                            if self.inputType == 'TSS':
                                target = self.check_if_contains_TSS(chrom, start, start + tokenized_sequence_len)
                            elif self.inputType == 'CRE':
                                #target = self.check_if_cre_enriched(chrom, start, end)
                                target = self.check_if_exceeds_min_coverage(chrom, start, start + tokenized_sequence_len)
                            
                            start += tokenized_sequence_len
                            
                            
                            interval_group = h5f.create_group(str(idx))
                            interval_group.create_dataset('input_ids', data=np.array(tokens, dtype=np.int32))
                            interval_group.attrs['target'] = target
                            interval_group.attrs['start'] = start
                            interval_group.attrs['end'] = end
                            interval_group.attrs['chrom'] = chrom
                            interval_group.attrs['taxon'] = self.taxon
                            

                            if idx % 100 == 0:
                                h5f.flush()
                                
                            #pbar.update()
                            
                            idx += 1
                #pbar.close()
                h5f.flush()
            
            os.rename(temp_path, self.h5_cache_path)
            self.h5_cache = h5py.File(self.h5_cache_path, 'r')
            
        except Exception as e:
            self.logger.error(f'Error creating cache: {e}')
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    def __getitem__(self, idx):
        grp = self.h5_cache[str(idx)]
        token_ids = grp['input_ids'][()]
        
        target = grp.attrs['target']
        taxon = random.choices([grp.attrs['taxon'], 0], weights = [1-self.unknown_taxon_prob, self.unknown_taxon_prob], k=1)[0]
        
        seq_ids = [self.cls_id] + token_ids.tolist() + [self.sep_id]
        seq_len = len(seq_ids)
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            seq_ids += [self.pad_id] * pad_len
            attn_mask = [1] * seq_len + [0] * pad_len
        else:
            attn_mask = [1] * self.max_seq_len
            
        input_ids = torch.tensor(seq_ids, dtype = torch.long)
        attention_mask = torch.tensor(attn_mask, dtype = torch.long)
        
        labels = torch.zeros(self.max_seq_len, 1, dtype=torch.float32)
        labels_mask = torch.zeros(self.max_seq_len, 1, dtype = torch.bool)
        
        labels[0, 0] = float(target)
        labels_mask[0, 0] = True
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'labels_mask': labels_mask,
            'taxon': taxon,
        }
        
        
    def __del__(self):
        try:
            if hasattr(self, 'h5_cache') and self.h5_cache is not None:
                self.h5_cache.close()
        except Exception:
            pass
        
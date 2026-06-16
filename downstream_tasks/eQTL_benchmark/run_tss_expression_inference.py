#!/usr/bin/env python3
"""Run ExpressionCounts inference on 9kbp TSS-centered gene regions.

Maps unique genes to their true chromosome via Liver.train.v2.csv, extracts 
centered 9kbp sequences in-memory, and runs direct model inference.
Skips and logs genes missing required TSS positions or chromosome mappings.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import csv
import pandas as pd
from pysam import FastaFile
import json
import re
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from downstream_tasks.expression_prediction.expression_model_final import ExpressionCounts

def load_description_text(meta_path: str | Path) -> str:
    """Reads a metadata JSON file and parses it into a unified text description."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    line_texts = []
    for k, v in meta.items():
        k = str(k).replace('_', ' ')
        v = str(v).replace('_', ' ')
        clean_k = re.sub(r'^(Characteristics|Chracteristics|Charateristics|Parameter)\s*', '', k)
        clean_k = re.sub(r'\[|\]', '', clean_k).strip()
        clean_k = clean_k if clean_k else k
        clean_v = str(v).replace('"', '').strip()
        line_texts.append(f'{clean_k} is {clean_v}.')
        
    if not line_texts:
        raise ValueError(f"No description text could be generated from file: {meta_path}")
    return " ".join(line_texts)

# --- Dataset Processing Pipeline ---

class GeneTssInferenceDataset(Dataset):
    """Prepares 9kbp genomic sequences for the model with strict token length checks."""
    
    def __init__(
        self, 
        genome: str | Path, 
        dna_tokenizer, 
        text_tokenizer, 
        description: str,
        genes_path: str,
        seed: int = 42,
        num_before: int = 512,
        gen_max_seq_len: int = 1024,
        token_len_for_fetch: int = 10,
    ):
        self.genome = genome
        self.sequences = FastaFile(self.genome)
        self.gen_tokenizer = dna_tokenizer
        self.text_tokenizer = text_tokenizer
        self.description = description

        self.genes = pd.read_csv(genes_path, sep='\t')
        self.genes["strand"] = "+"

        self.logger = logging.getLogger(__name__)
        
        self.num_before = num_before
        self.token_len_for_fetch = token_len_for_fetch
        self.seed = seed
        np.random.seed(self.seed)
        self.gen_max_seq_len = gen_max_seq_len

        self.n_keys = 1 #assuming one target

    def _ensure_sequences_open(self, required_for: str = "sequence access"):
        if self.sequences is not None:
            return
        if not self.genome or not os.path.exists(self.genome):
            raise FileNotFoundError(
                f"Genome fasta is required for {required_for}, but was not found: {self.genome}"
            )
        self.sequences = FastaFile(self.genome)
    
    def reverse_complement(self, sequence):
        complement = str.maketrans('ACGTN', 'TGCAN')
        return sequence.translate(complement)[::-1]
    
    def tokenize_genome(self, i):
        self._ensure_sequences_open("genome tokenization")
        row = self.genes.iloc[i]  
        chrom = row["chromosome"] 
        start = row["TSS"]
        end = row["TES"] 
        strand = row["strand"]
        gene_id = row["gene_id_unversioned"]
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
        return start_gene, token_lengths_df, gene_id
    

    def __len__(self) -> int:
        return self.genes.shape[0]

    def __getitem__(self, idx: int) -> dict:
        
        _, tokens_df, gene_id = self.tokenize_genome(idx)

        cls_id = self.gen_tokenizer.cls_token_id
        sep_id = self.gen_tokenizer.sep_token_id

        assert (cls_id is not None) and (sep_id is not None), "Tokenizer must have CLS/SEP"
        input_ids = tokens_df["token_id"].values.astype(np.int32)
        L = min(input_ids.shape[0], self.gen_max_seq_len - 2)
        input_ids = input_ids[:L]
        
        tok = torch.as_tensor(input_ids, dtype=torch.long)
        seq_input_ids = torch.cat([tok.new_tensor([cls_id]), tok, tok.new_tensor([sep_id])], dim=0)
        assert seq_input_ids.size(0) <= 1024, f"Tokenized sequence length with CLS/SEP must less then 1024, but got {seq_input_ids.size(0)}"
        seq_attn_mask = torch.ones(seq_input_ids.size(0), dtype=torch.long)
        seq_token_types = torch.zeros(seq_input_ids.size(0), dtype=torch.long)
        
        batch_input_ids   = seq_input_ids.unsqueeze(0).expand(self.n_keys, -1)
        batch_attn_mask   = seq_attn_mask.unsqueeze(0).expand(self.n_keys, -1)
        batch_token_types = seq_token_types.unsqueeze(0).expand(self.n_keys, -1)

        labels = torch.zeros((self.n_keys, L + 2, 1), dtype=torch.float32)
        labels_mask = torch.zeros((self.n_keys, L + 2, 1), dtype=torch.bool)

        tokenized_desc = self.text_tokenizer(
            [self.description], 
            truncation=True, 
            padding="max_length", 
            max_length=510, 
            return_tensors="pt"
        )

        desc_ids = tokenized_desc["input_ids"].reshape(1,  tokenized_desc["input_ids"].shape[1])
        desc_msk = tokenized_desc["attention_mask"].reshape(1,  tokenized_desc["attention_mask"].shape[1])

        features = {
            "gene_id": gene_id,
            "input_ids": batch_input_ids,          
            "attention_mask": batch_attn_mask,    
            "token_type_ids": batch_token_types,                 
            "dataset_flag": torch.ones(self.n_keys, dtype=torch.float32),
            "desc_input_ids": desc_ids,         
            "desc_attention_mask": desc_msk
        }
        return features
    
class CollateFn:
    tokenizer = None
    text_tokenizer = None
    @staticmethod
    def _pad_1d(x: torch.Tensor, length: int, pad_value: int, pad_left: bool = False) -> torch.Tensor:
        pad_len = length - x.size(0)
        if pad_len <= 0:
            return x
        pad = x.new_full((pad_len,), pad_value)
        return torch.cat([pad, x], dim=0) if pad_left else torch.cat([x, pad], dim=0)

    @staticmethod
    def _pad_2d(x: torch.Tensor, max_len: int, pad_value, dim: int = 0) -> torch.Tensor:
        pad_len = max_len - x.size(dim)
        if pad_len <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        pad = x.new_full(tuple(pad_shape), pad_value)
        return torch.cat([x, pad], dim=dim)

    @staticmethod
    def _pad_3d(x: torch.Tensor, max_len: int, pad_value, dim: int = 1) -> torch.Tensor:
        pad_len = max_len - x.size(dim)
        if pad_len <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        pad = x.new_full(tuple(pad_shape), pad_value)
        return torch.cat([x, pad], dim=dim)

    @classmethod
    def collate_fn(cls, batch):
        pad_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        no_pad_keys = ['dataset_flag']
        special_keys = ['gene_id']

        pad_token_ids = {
            'input_ids': cls.tokenizer.pad_token_id,
            'attention_mask': 0,
            'token_type_ids': 0,
            'labels': 0.0,
            'labels_mask': 0,
            'desc_input_ids': cls.text_tokenizer.pad_token_id,
            'desc_attention_mask': 0,
        }

        max_seq_len = max(sample['input_ids'].size(1) for sample in batch)
        assert max_seq_len <= 1024, f"Max sequence length in batch exceeds 1024: {max_seq_len}"
        max_seq_len=1024
        n_keys = len(batch[0]['desc_input_ids'])
        max_text_seq_len = 0
        for sample in batch:
            for ids in sample['desc_input_ids']:
                max_text_seq_len = max(max_text_seq_len, ids.size(0))
        if max_text_seq_len == 0:
            max_text_seq_len = 1

        batch_dict = {key: [] for key in pad_keys + no_pad_keys + special_keys}

        desc_ids_batch = []
        desc_mask_batch = []
        for sample in batch:
            sample_ids = []
            sample_masks = []
            for k in range(n_keys):
                ids = sample['desc_input_ids'][k]       # (L_i,)
                mask = sample['desc_attention_mask'][k] # (L_i,)

                ids  = CollateFn._pad_1d(ids,  max_text_seq_len, pad_token_ids['desc_input_ids'], pad_left=True)
                mask = CollateFn._pad_1d(mask, max_text_seq_len, pad_token_ids['desc_attention_mask'], pad_left=True)

                sample_ids.append(ids)
                sample_masks.append(mask)

            desc_ids_batch.append(torch.stack(sample_ids, dim=0))   # (n_keys, L_text_max)
            desc_mask_batch.append(torch.stack(sample_masks, dim=0))# (n_keys, L_text_max)

        for sample in batch:
            for key in pad_keys:
                x = sample[key]
                if key in ['input_ids', 'attention_mask', 'token_type_ids']:  # (n_keys, L)
                    x = CollateFn._pad_2d(x, max_seq_len, pad_token_ids[key], dim=1)    # pad по L -> dim=1
                if key in ['labels', 'labels_mask']:                                                       
                    x = CollateFn._pad_3d(x, max_seq_len, pad_token_ids[key], dim=1)
                batch_dict[key].append(x)

            for key in no_pad_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])

            for key in special_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])

        # stack
        for key in pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)  # (B, n_keys, Lmax) или (B, n_keys, Lmax, 1)

        for key in no_pad_keys:
            if len(batch_dict[key]) > 0:
                batch_dict[key] = torch.stack(batch_dict[key], dim=0)

        batch_dict['desc_input_ids'] = torch.stack(desc_ids_batch, dim=0)        # (B, n_keys, L_text_max)
        batch_dict['desc_attention_mask'] = torch.stack(desc_mask_batch, dim=0)  # (B, n_keys, L_text_max)

        return batch_dict

# --- Model Evaluation Execution Engine ---

def run_tss_inference(model, dataloader, device, use_amp: bool) -> dict[int, float]:
    """Runs a single forward pass over extracted sequence batches."""
    results = {}
    autocast_enabled = bool(use_amp and device.type == "cuda")
    
    with torch.no_grad():
        for batch in dataloader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                output = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device), 
                        desc_input_ids=batch["desc_input_ids"].to(device), 
                        desc_attention_mask=batch["desc_attention_mask"].to(device), 
                        dataset_flag=batch["dataset_flag"].to(device))
            
            logits = output["logits"][:, 0, 0].float().cpu().tolist()
            for gene_id, val in zip(batch["gene_id"], logits):
                results[gene_id] = val
                        
    return results


# --- Main Orchestration Loop ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liver-train-csv", default="data/Attachments_robert_2/Liver.train.v2.csv")
    parser.add_argument("--gene-tss-csv", default="data/Attachments_robert_2/gene_tss.v2.csv")
    parser.add_argument("--b37-fasta", required=True, help="Path to hg19/B37 reference FASTA database")
    parser.add_argument("--out", required=True, help="Output destination filepath (.csv or .tsv)")
    parser.add_argument("--log-skipped", default="skipped_genes.log", help="File destination for logs regarding skipped genes")
    parser.add_argument("--config", default="notebooks/inference.yaml")
    parser.add_argument("--gena-lm-home", required=True, help="Path to GENA_LM project workspace environment")
    parser.add_argument("--checkpoint", required=True, help="Path to weights file checkpoint binary")
    parser.add_argument("--description-path", required=True, help="Metadata JSON description text configuration file")
    parser.add_argument("--description-name", default="HepG2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    # Environment runtime initialization setup
    os.environ["GENALM_HOME"] = str(Path(args.gena_lm_home).expanduser().resolve())
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else ("cpu" if args.device == "auto" else args.device))
    
    experiment_config_path = Path(args.config).expanduser().absolute()
    with initialize_config_dir(str(experiment_config_path.parents[0])):
        experiment_config = compose(config_name=experiment_config_path.name)

    model_kwargs = instantiate(experiment_config["model_kwargs"])
    # initialize model
    model = ExpressionCounts(**model_kwargs)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    # Architecture and Tokenizer instantiation steps
    dna_tok = AutoTokenizer.from_pretrained(experiment_config["args_params"]["gen_tokenizer"])
    text_tok = AutoTokenizer.from_pretrained(experiment_config["shared_dataset_params"]["text_tokenizer"], padding_side="left")
    CollateFn.tokenizer = dna_tok
    CollateFn.text_tokenizer = text_tok
    description_text = load_description_text(args.description_path)

    # Dataset & PyTorch DataLoader setup
    dataset = GeneTssInferenceDataset(
        genome = args.b37_fasta, 
        dna_tokenizer = dna_tok, 
        text_tokenizer = text_tok, 
        description = description_text,
        genes_path = args.liver_train_csv)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=CollateFn.collate_fn)

    predictions = run_tss_inference(model, dataloader, device, use_amp=not args.no_amp)

    pd.DataFrame(list(predictions.items()), columns=['gene_id', 'predicted_expression']).to_csv(args.out)

    print(f"Pipeline executed successfully. Target predictions saved into: {args.out}")


if __name__ == "__main__":
    main()
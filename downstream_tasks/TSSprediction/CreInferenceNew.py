import logging
import os
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator
import torch
import torch.utils.data as data
from transformers import AutoTokenizer
from pysam import FastaFile
import polars as pl
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from safetensors.torch import load_file
import numpy as np
from tqdm import tqdm

_fasta_cache = {}

def get_fasta_handle(path: str):
    """Return a FastaFile handle for the given path, reusing it in the same process."""
    if path not in _fasta_cache:
        _fasta_cache[path] = FastaFile(path)
    return _fasta_cache[path]


class CreDataset(data.IterableDataset):
    def __init__(self,
                 genome_list: List[Tuple[str, str]],  # (genome_path, taxon)
                 tokenizer_path: str,
                 max_seq_len: int = 1024,
                 token_len_for_fetch: int = 10):
        super().__init__()
        self.genome_list = genome_list
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.token_len_for_fetch = token_len_for_fetch
        self.num_special_tokens = 2
        self.interval_len_tokens = max_seq_len - self.num_special_tokens
        self.cls_id = None
        self.sep_id = None
        self.pad_id = None

        self.taxon_mapping = {'Unknown': 0,
                    "Lepidosauria": 1,
                    "Chondrichthyes": 2,
                    "Mammalia": 3,
                    "Amphibia": 4,
                    "Actinopteri": 5,
                    "Myxini": 6,
                    "Aves": 7}

    def encode_taxon(self, taxon: str) -> int:
        return self.taxon_mapping.get(taxon, 0)

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

    def _generate_windows(self, genome_list):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        for genome_idx, (genome_path, taxon) in enumerate(genome_list):
            taxon_encoded = self.encode_taxon(taxon)
            fasta = get_fasta_handle(genome_path)
            for chrom in fasta.references:
                chrom_len = fasta.get_reference_length(chrom)
                start = 0
                while start < chrom_len:
                    end = min(start + self.interval_len_tokens * self.token_len_for_fetch, chrom_len)
                    sequence = fasta.fetch(chrom, start, end).upper()

                    encoded = tokenizer.encode_plus(sequence, return_offsets_mapping=True)
                    tokens = encoded['input_ids'][1:-1]
                    mapping = encoded['offset_mapping'][1:-1]

                    tokens = tokens[:self.interval_len_tokens]
                    mapping = mapping[:self.interval_len_tokens]

                    tokenized_len = self.compute_encoded_seq_len(tokens, mapping)

                    seq_ids = [self.cls_id] + tokens + [self.sep_id]
                    seq_len = len(seq_ids)
                    if seq_len < self.max_seq_len:
                        pad_len = self.max_seq_len - seq_len
                        seq_ids += [self.pad_id] * pad_len
                        attn_mask = [1] * seq_len + [0] * pad_len
                    else:
                        attn_mask = [1] * self.max_seq_len

                    yield {
                        'input_ids': torch.tensor(seq_ids, dtype=torch.long),
                        'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
                        'taxon': torch.tensor(taxon_encoded, dtype=torch.long),
                        'genome_idx': genome_idx,
                        'chrom': chrom,
                        'start': start,
                        'end': start+tokenized_len,
                    }
                    start += tokenized_len

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:

            yield from self._generate_windows(self.genome_list)
            
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            my_genomes = [g for i, g in enumerate(self.genome_list) if i % num_workers == worker_id]
            yield from self._generate_windows(my_genomes)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    taxon = torch.stack([item['taxon'] for item in batch]) 
    genome_idx = [item['genome_idx'] for item in batch]
    chrom = [item['chrom'] for item in batch]
    start = [item['start'] for item in batch]
    end = [item['end'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'taxon': taxon,
        'genome_idx': genome_idx,
        'chrom': chrom,
        'start': start,
        'end': end,
    }

class CreInference:
    def __init__(self,
                 checkpoint: str,
                 tokenizer_path: str,
                 config: str,
                 output_dir: str,
                 genome_list: List[Tuple[str, str]],
                 max_seq_len: int = 1024,
                 token_len_for_fetch: int = 10,
                 batch_size: int = 4,
                 num_workers: int = 8,
                 device: str = 'cuda'):
        self.checkpoint = checkpoint
        self.config = config
        self.output_dir = output_dir
        self.genome_list = genome_list
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.token_len_for_fetch = token_len_for_fetch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = None
        self.logger = logging.getLogger(__name__)

    def instantiate_model(self):
        self.logger.info(f'Loading model on {self.device}')
        config_path = Path(self.config).absolute()
        with initialize_config_dir(str(config_path.parents[0])):
            experiment_config = compose(config_name=config_path.name)
        model = instantiate(experiment_config['model'])
        state_dict = load_file(self.checkpoint, device='cpu')
        model.load_state_dict(state_dict)
        self.model = model.to(self.device)
        self.model.eval()

    def run(self):
        self.instantiate_model()
        os.makedirs(self.output_dir, exist_ok=True)

        for genome_idx, (genome_path, taxon) in enumerate(self.genome_list):
            self.logger.info(f'Processing {genome_path} on {self.device}')

            single_genome_list = [(genome_path, taxon)]
            dataset = CreDataset(
                genome_list=single_genome_list,
                tokenizer_path=self.tokenizer_path,
                max_seq_len=self.max_seq_len,
                token_len_for_fetch=self.token_len_for_fetch
            )

            dataloader = data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                pin_memory=True
            )

            results = []

            with torch.inference_mode():
                for batch in tqdm(dataloader, desc=f"GPU {self.device[-1]} - {Path(genome_path).stem}", unit="batch"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    taxon_tensor = batch['taxon'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels_mask=None,
                        labels=None,
                        taxon=taxon_tensor.unsqueeze(1)
                    )

                    cls_logits = outputs.logits[:, 0, :] 
                    probs = torch.sigmoid(cls_logits).cpu().numpy().flatten()

                    for chrom, start, end, prob in zip(
                        batch['chrom'], batch['start'], batch['end'], probs
                    ):
                        results.append((chrom, start, end, float(prob)))


            out_path = os.path.join(self.output_dir, Path(genome_path).stem + '.csv')
            if results:
                df = pl.DataFrame(results, schema=['chrom', 'start', 'end', 'cre_prob'])
                df.write_csv(out_path)
                self.logger.info(f'Written {out_path} ({len(results)} windows)')
            else:
                self.logger.warning(f'No windows for {genome_path}')


def gpu_worker(chunk: List[Tuple[str, str]], gpu_id: int,
               checkpoint: str, tokenizer_path: str, config: str,
               output_dir: str, batch_size: int, num_workers: int):
    """Process a chunk of genomes on a specific GPU."""
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    predictor = CreInference(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        config=config,
        output_dir=output_dir,
        genome_list=chunk,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )
    predictor.run()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    CHECKPOINT = '/workspace-SR003.nfs2/estsoi/TSSprediction/runs/CRE_prediction/checkpoint-31750/model.safetensors'
    TOKENIZER_PATH = '/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/data/tokenizers/t2t_1000h_multi_32k'
    CONFIG_PATH = '/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/configs/cre_test_config.yaml'
    OUTPUT_DIR = '/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/predicts'
    MAPPINGS_CSV = '/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/metadata/genome2taxon.csv'

    BATCH_SIZE = 60 
    NUM_WORKERS = 30 
    NUM_GPUS = 7

    # Load all genomes
    df = pl.read_csv(MAPPINGS_CSV, has_header=True)
    genomes_unfiltered = [(row['genomePath'], row['taxon']) for row in df.iter_rows(named=True)]
    all_genomes = []
    genomes_DONE = [] 
    for pair in genomes_unfiltered:
        genome_path = pair[0]
        out_path = os.path.join(OUTPUT_DIR, Path(genome_path).stem + '.csv')
        if os.path.exists(out_path):
            genomes_DONE.append(pair)
        else:
            all_genomes.append(pair)
        
    logger.info(f'Total genomes: {len(genomes_unfiltered)}')
    logger.info(f'Genomes already completed: {len(genomes_DONE)}')
    logger.info(f'Genomes to be completed: {len(all_genomes)}')

    # Split into NUM_GPUS chunks (round‑robin)
    chunks = [[] for _ in range(NUM_GPUS)]
    for i, genome in enumerate(all_genomes):
        chunks[i % NUM_GPUS].append(genome)

    # Launch one process per GPU
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = mp.Process(target=gpu_worker,
                       args=(chunk, gpu_id, CHECKPOINT, TOKENIZER_PATH, CONFIG_PATH,
                             OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    logger.info('All inference jobs finished.')
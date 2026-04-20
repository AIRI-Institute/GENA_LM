import logging
import torch
from pysam import FastaFile
from pathlib import Path
from transformers import AutoTokenizer

import polars as pl

import logging
from pathlib import Path
from hydra.utils import instantiate
import torch
from hydra import initialize_config_dir, compose

from safetensors.torch import load_file
import os
import multiprocessing as mp 


checkpoint = '/workspace-SR003.nfs2/estsoi/TSSprediction/runs/CRE_prediction/checkpoint-31750/model.safetensors'


def _process_genome_chunk(chunk, gpu_id, checkpoint, tokenizer_path, config, output_dir,
                          max_seq_len, token_len_for_fetch):
    """Worker function for a single GPU process."""
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    
    predictor = CreInference(
        checkpoint=checkpoint,
        tokenizer=tokenizer_path,
        config=config,
        output_dir=output_dir,
        max_seq_len=max_seq_len,
        token_len_for_fetch=token_len_for_fetch,
        device=device
    )
    predictor.instantiate_model()
    for genomePath, taxon in chunk:
        predictor.process_single_genome(genomePath, taxon)

class CreInference():
    
    def __init__(self, checkpoint:str, 
                    tokenizer:str, 
                    config:str,
                    output_dir:str,
                    genome:str|None = None,
                    taxon:str|None = None,
                    mappings:str|None = None,
                    max_seq_len:int = 1024,
                    token_len_for_fetch:int = 10,
                    n_processes:int=20,
                    device:str|None = None
                    ):
        self.checkpoint = checkpoint
        self.config = config
        self.output_dir = output_dir
        if genome and taxon:
            self.genome = genome
            self.taxon = taxon
            self.single_input = True
        if mappings:
            self.mappings = pl.read_csv(mappings, has_header=True)
            self.single_input = False
        self.model = None
        self.tokenizer_path = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        
        self.token_len_for_fetch = token_len_for_fetch
        
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        
        self.num_special_tokens = 2
        self.max_seq_len = max_seq_len
        self.interval_len_tokens = self.max_seq_len - self.num_special_tokens
        
        self.n_processes = n_processes
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)   
        
    def encode_taxon(self, taxon) -> int:
        mapping = {'Unknown': 0,
                    "Lepidosauria": 1,
                    "Chondrichthyes": 2,
                    "Mammalia": 3,
                    "Amphibia": 4,
                    "Actinopteri": 5,
                    "Myxini": 6,
                    "Aves": 7}
        return mapping[taxon] if taxon in mapping else mapping['Unknown']
             
    def instantiate_model(self):
        self.logger.info(f'INSTANTIATING MODEL on {self.device}')
        if not isinstance(self.config, Path):
            self.config = Path(self.config).absolute()
        with initialize_config_dir(str(self.config.parents[0])):
            experiment_config = compose(config_name=self.config.name)
            self.config_dict = experiment_config
        model = instantiate(experiment_config['model'])
        
        state_dict = load_file(checkpoint, device="cpu")
        model.load_state_dict(state_dict)
        self.model = model
        self.model.to(self.device)
        
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
        
    def process_single_genome(self, genomePath:str, taxon:str):
        self.logger.info(f'Processing genome {genomePath}, taxon: {taxon} on {self.device}')
        
        assert isinstance(self.model, torch.nn.Module)
        self.model.eval()
        
        with torch.inference_mode():
            results = []
            with FastaFile(genomePath) as genome:
                for chrom in genome.references:
                    chrom_results = []
                    chrom_len = genome.get_reference_length(chrom)
                    start = 0
                    while start < chrom_len:
                        end = min(start+self.interval_len_tokens*self.token_len_for_fetch, chrom_len)
                        sequence = genome.fetch(chrom, start, end).upper()
                        
                        encoded_sequence = self.tokenizer.encode_plus(sequence, return_offsets_mapping=True)
                        encoded_sequence['input_ids'] = encoded_sequence['input_ids'][1:-1]
                        encoded_sequence['offset_mapping'] = encoded_sequence['offset_mapping'][1:-1]
                        
                        if encoded_sequence['input_ids'].__len__() < self.interval_len_tokens:
                            pass
                            
                        mapping = encoded_sequence['offset_mapping'][0:self.interval_len_tokens]
                        tokens = encoded_sequence['input_ids'][0:self.interval_len_tokens]
                        
                        tokenized_sequence_len = self.compute_encoded_seq_len(tokens, mapping)
                        
                        taxon_encoded = torch.tensor(self.encode_taxon(taxon)).unsqueeze(0).to(self.device)
                        
                        seq_ids = [self.cls_id] + tokens + [self.sep_id]
                        seq_len = len(seq_ids)
                        if seq_len < self.max_seq_len:
                            pad_len = self.max_seq_len - seq_len
                            seq_ids += [self.pad_id] * pad_len
                            attn_mask = [1] * seq_len + [0] * pad_len
                        else:
                            attn_mask = [1] * self.max_seq_len
                            
                        input_ids = torch.tensor(seq_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                        attention_mask = torch.tensor(attn_mask, dtype=torch.long).unsqueeze(0).to(self.device)
                        
                        labels = torch.zeros(self.max_seq_len, 1, dtype=torch.float32).unsqueeze(0).to(self.device)
                        labels_mask = torch.zeros(self.max_seq_len, 1, dtype=torch.bool).unsqueeze(0).to(self.device)
                        
                        labels[0, 0] = 0.0
                        labels_mask[0, 0] = True
                        
                        model_output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels_mask=labels_mask,
                            labels=labels,
                            taxon=taxon_encoded
                        )
                        
                        predict_raw = model_output.predicts[:, 0:1, :].unsqueeze(-1)
                        predict = torch.sigmoid(predict_raw).cpu().numpy().flatten()
                        
                        region_result = (chrom, start, start+tokenized_sequence_len, predict.item(), taxon_encoded)
                        chrom_results.append(region_result)
                        
                        start += tokenized_sequence_len
                        
                    results += chrom_results
            
            out_path = os.path.join(
                self.output_dir,
                os.path.basename(genomePath).removesuffix('.fasta') + '.csv'
            )
            pl.DataFrame(results, schema=('chrom', 'start', 'end', 'cre_prob', 'taxon')).write_csv(out_path)
            self.logger.info(f'Finished {genomePath} -> {out_path}')
                        
    def run(self, num_gpus: int = 1):
        
        if self.single_input:
            self.instantiate_model()
            self.process_single_genome(self.genome, self.taxon)
            return

        tasks = [(row['genomePath'], row['taxon']) for row in self.mappings.iter_rows(named=True)]


        if num_gpus <= 1 or len(tasks) == 1:
            self.instantiate_model()
            for genomePath, taxon in tasks:
                self.process_single_genome(genomePath, taxon)
            return

        available_gpus = torch.cuda.device_count()
        num_gpus = min(num_gpus, available_gpus, len(tasks))
        self.logger.info(f"Using {num_gpus} GPUs for {len(tasks)} genomes")

        chunks = [[] for _ in range(num_gpus)]
        for i, task in enumerate(tasks):
            chunks[i % num_gpus].append(task)

        args_list = [
            (chunks[gpu_id], gpu_id, self.checkpoint, self.tokenizer_path, self.config,
            self.output_dir, self.max_seq_len, self.token_len_for_fetch)
            for gpu_id in range(num_gpus) if chunks[gpu_id]
        ]
        mp.set_start_method('spawn', force=True)
        processes = []
        for args in args_list:
            p = mp.Process(target=_process_genome_chunk, args=args)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()   # optional on Linux, required for Windows/frozen exe

    tokenizer = '/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/data/tokenizers/t2t_1000h_multi_32k'
    predictor = CreInference(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        config='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/configs/cre_test_config.yaml',
        output_dir='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/predicts',
        mappings='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/metadata/genome2taxon.csv'
    )
    predictor.instantiate_model()
    predictor.run(num_gpus=torch.cuda.device_count())
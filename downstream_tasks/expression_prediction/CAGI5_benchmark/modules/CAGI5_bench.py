import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from safetensors.torch import load_file
#import hydra
from transformers import AutoTokenizer
import tqdm
import re
import sys
from contextlib import contextmanager
from pathlib import Path
import importlib.util
from mpramnist.Kircher2019.dataset import KircherDataset
from DescriptionLookup import DescriptionLookup


class CAGI5_bench():
    
    def __init__(self, 
                    model_cls:str,
                    model_checkpoint:str,
                    model_config:str,
                    dna_tokenizer:str,
                    desc_tokenizer:str,
                    dna_max_seq_len:int,
                    desc_max_seq_len:int,
                    token_len_for_fetch:int,
                    num_before:int,
                    celltype2desc:dict|DescriptionLookup,
                    device:str = 'cuda:6' if torch.cuda.is_available() else 'cpu'
                    ):
        self.dataset = None
        self.logger = logging.getLogger(__name__)
        self.model_path, self.model_cls = model_cls.split('::')              #format : /path/to/python/file.py::class_name
        self.model_path : Path = Path(self.model_path).resolve()
        self.model_config = model_config
        self.model_checkpoint = model_checkpoint
        self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_tokenizer)
        self.cls_id = self.dna_tokenizer.cls_token_id
        self.sep_id = self.dna_tokenizer.sep_token_id
        self.pad_id = self.dna_tokenizer.pad_token_id
        
        self.desc_tokenizer = AutoTokenizer.from_pretrained(desc_tokenizer, padding_side="right")
        self.token_len_for_fetch = token_len_for_fetch
        self.dna_max_seq_len = dna_max_seq_len
        self.dna_max_seq_tokens = dna_max_seq_len - 2
        self.desc_max_seq_len = desc_max_seq_len
        self.desc_max_seq_tokens = desc_max_seq_len - 2
        self.device = device
        self.num_before = num_before
        self.celltype2desc = celltype2desc
        
    
    def initialize_dataset(self, 
                            elements:list|None = None, 
                            cell_type:list|None = None,
                            transform=None, 
                            target_transform=None, 
                            root=None,
                            context_fn = lambda seq, el: seq,
                            padding_fn = lambda seq: seq,
                            mode:str = 'default',
                            length:int = 200,
                            max_pvalue:float|None=None,
                            min_tags:float|None=None):
        
        def _make_transform(context_fn = context_fn, padding_fn = padding_fn):
            def transform(sequence:str, element:str):
                sequence_in_context = context_fn(sequence, element)
                sequence_in_context_padded = padding_fn(sequence_in_context)
                return self.tokenize_sequence_centered(sequence_in_context_padded)
            return transform
                
        self.dataset = KircherDataset(elements=elements, length=length, mode=mode, max_pvalue=max_pvalue, min_tags=min_tags, cell_type=cell_type, transform=_make_transform() if transform is None else transform, target_transform=target_transform, root=root)
        
    def make_dataloader(self, 
                        batch_size:int, 
                        n_workers:int):
        def make_collate_fn(dataset_flag:int = 0):
            
            def collate_fn(batch):
                #batch_descriptions = [self.make_description_from_json(item[2]) for item in batch] #item[2] - cell type
                batch_descriptions = [self.make_description_from_json(
                                                                    self.celltype2desc[item[2]]
                                                                            ) for item in batch]
                
                batch_tokenized_descriptions = [self.tokenize_description(desc) for desc in batch_descriptions]
                
                batched = {
                    'dna_input_ids_wt': torch.stack([item[0]['seq']['dna_input_ids'] for item in batch]),
                    'dna_input_ids_alt': torch.stack([item[0]['seq_alt']['dna_input_ids'] for item in batch]),
                    'dna_attention_mask_wt': torch.stack([item[0]['seq']['dna_attention_mask'] for item in batch]),
                    'dna_attention_mask_alt': torch.stack([item[0]['seq_alt']['dna_attention_mask'] for item in batch]),
                    'difference': torch.stack([item[1].clone().detach() for item in batch]),
                    'desc_input_ids': torch.stack([desc_encoded['desc_input_ids'] for desc_encoded in batch_tokenized_descriptions]),
                    'desc_attention_mask': torch.stack([desc_encoded['desc_attention_mask'] for desc_encoded in batch_tokenized_descriptions]),
                    'dataset_flag': torch.stack([torch.tensor(dataset_flag) for _ in range(len(batch))]).unsqueeze(-1),
                    'element': [item[3] for item in batch]
                }
                return batched
            return collate_fn
        
        
        self.dataloader = DataLoader(dataset = self.dataset, collate_fn=make_collate_fn(), batch_size=batch_size, num_workers=n_workers, shuffle=False)
        
    #def initialize_model(self):
    #    self.logger.info(f'Loading model on {self.device}')
    #    cls = hydra.utils.get_class(self.model_cls)
    #    config_path = Path(self.model_config)
    #    with initialize_config_dir(str(config_path.parents[0])):
    #        experiment_config = compose(config_name=config_path.name)
    #    model_kwargs = instantiate(experiment_config['model_kwargs'])
    #    model = cls(**model_kwargs)
    #    if Path(self.model_checkpoint).suffix == '.tensors':
    #        state_dict = load_file(self.model_checkpoint, device='cpu')
    #    else:
    #        state_dict = torch.load(self.model_checkpoint, map_location=torch.device('cpu'), weights_only=True)
    #    model.load_state_dict(state_dict)
    #    self.model = model.to(self.device)
    #    self.model.eval()
    
    def initialize_model(self):
        
        @contextmanager
        def _temporary_sys_path(path: Path):
            path = str(path)
            added = path not in sys.path
            if added:
                sys.path.insert(0, path)
            try:
                yield
            finally:
                if added:
                    sys.path.remove(path)
        
        self.logger.info(f'Loading model on {self.device}')
        
        module_name = f"_loaded_model_{self.model_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, self.model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {self.model_path}")

        module = importlib.util.module_from_spec(spec)
        
        with _temporary_sys_path(path = self.model_path.parent):
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        try:
            cls = getattr(module, self.model_cls)
        except AttributeError as exc:
            raise ImportError(f"{self.model_cls!r} not found in {self.model_path}") from exc
        
        config_path = Path(self.model_config)
        with initialize_config_dir(str(config_path.parents[0])):
            experiment_config = compose(config_name=config_path.name)
        model_kwargs = instantiate(experiment_config['model_kwargs'])
        model = cls(**model_kwargs)
        if Path(self.model_checkpoint).suffix == '.tensors':
            state_dict = load_file(self.model_checkpoint, device='cpu')
        else:
            state_dict = torch.load(self.model_checkpoint, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device)
        self.model.eval()
    
    
    def reverse_complement(self, sequence):
        complement = str.maketrans('ACGTN', 'TGCAN')
        return sequence.translate(complement)[::-1]


    def tokenize_sequence_centered(self, sequence: str, strand: str = "+"):
        """
        Tokenize a DNA sequence centered on TSS.
        Returns a flat list of token IDs with CLS at start and SEP at end.
        """
        sequence = sequence.upper()
        
        reverse = (strand == "-")
        tss_pos = len(sequence) // 2  # TSS always in the middle
        tokens = []

        # 1. UPSTREAM (5' side of TSS)
        if self.num_before > 0:
            fetch_len = self.num_before * self.token_len_for_fetch
            if not reverse:
                up_seq = sequence[max(0, tss_pos - fetch_len):tss_pos]
            else:
                up_seq = sequence[tss_pos:min(len(sequence), tss_pos + fetch_len)]
                up_seq = self.reverse_complement(up_seq)
                
            enc = self.dna_tokenizer.encode_plus(up_seq, return_offsets_mapping=False)
            up_tokens = enc['input_ids'][1:-1]  # strip auto-added CLS/SEP
            
            #if len(up_tokens) < self.num_before:
            #    self.logger.warning(f"Upstream too short: {len(up_tokens)} < {self.num_before}")
                
            tokens.extend(up_tokens[-self.num_before:])

        # 2. DOWNSTREAM (3' side of TSS)
        if not reverse:
            down_seq = sequence[tss_pos:]
        else:
            down_seq = sequence[:tss_pos]
            down_seq = self.reverse_complement(down_seq)
            
        enc = self.dna_tokenizer.encode_plus(down_seq, return_offsets_mapping=False)
        tokens.extend(enc['input_ids'][1:-1][0:self.dna_max_seq_tokens-self.num_before])

        # 3. ORIENTATION & SPECIAL TOKENS
        if reverse:
            tokens.reverse()  # maintain 5' -> 3' transcriptional order
        
        input_ids =  [self.cls_id] + tokens + [self.sep_id]
        #if len(input_ids) != 1024:
        #    raise ValueError(f'len of input_ids is less then 1024 tokens: {len(input_ids)}')
        return {'dna_input_ids': torch.tensor(input_ids), 'dna_attention_mask': torch.tensor([1 for _ in range(len(input_ids))])}
    
    @staticmethod
    def make_description_from_json(meta):
        line_texts = []
        for k, v in meta.items():
            k = k.replace('_', ' ')
            v = str(v).replace('_', ' ')
            clean_k = re.sub(r'^(Characteristics|Chracteristics|Charateristics|Parameter)\\s*', '', k)
            clean_k = re.sub(r'\\[|\\]', '', clean_k).strip()
            clean_k = clean_k if clean_k else k
            clean_v = str(v).replace('"', '').strip()
            line_texts.append(f'{clean_k} is {clean_v}.')
        return " ".join(line_texts)
    
    
    def tokenize_description(self, description):
        
        #def _padding(input_ids, attention_mask):
        encoding = self.desc_tokenizer(description, 
                                        padding='max_length',
                                        padding_side='right',
                                        truncation=True,
                                        max_length=self.desc_max_seq_tokens,
                                        return_tensors="pt")
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        
        return {'desc_input_ids':input_ids, 'desc_attention_mask': attention_mask}
    
    
    def run_inference(self):

        with torch.autocast(device_type='cuda'), torch.inference_mode():

            true_vs_pred = {}

            for batch in tqdm.tqdm(self.dataloader):

                elements = batch['element']

                desc_input_ids = batch['desc_input_ids'].to(self.device)
                desc_attention_mask = batch['desc_attention_mask'].to(self.device)
                dataset_flag = batch['dataset_flag'].to(self.device, dtype=torch.bool)

                wt_model_output = self.model(
                    input_ids=batch['dna_input_ids_wt'].to(self.device),
                    attention_mask=batch['dna_attention_mask_wt'].to(self.device),
                    labels_mask=None,
                    labels=None,
                    return_dict=None,
                    desc_input_ids=desc_input_ids,
                    desc_attention_mask=desc_attention_mask,
                    dataset_flag=dataset_flag,
                )

                alt_model_output = self.model(
                    input_ids=batch['dna_input_ids_alt'].to(self.device),
                    attention_mask=batch['dna_attention_mask_alt'].to(self.device),
                    labels_mask=None,
                    labels=None,
                    return_dict=None,
                    desc_input_ids=desc_input_ids,
                    desc_attention_mask=desc_attention_mask,
                    dataset_flag=dataset_flag,
                )

                target_difference = batch['difference'].unsqueeze(-1).to(self.device)

                predicted_difference = (
                    alt_model_output.logits[:, 0:1, :]
                    - wt_model_output.logits[:, 0:1, :]
                ).squeeze(-1)

                # Shape: (batch_size, 2), columns are true and predicted difference.
                concat_tensor = torch.cat(
                    (target_difference, predicted_difference),
                    dim=1,
                ).float()

                for i, element in enumerate(elements):
                    element_tensor = concat_tensor[i:i + 1]

                    if element not in true_vs_pred:
                        true_vs_pred[element] = element_tensor
                    else:
                        true_vs_pred[element] = torch.cat(
                            (true_vs_pred[element], element_tensor),
                            dim=0,
                        )

            assert true_vs_pred is not None
            self.true_vs_pred = true_vs_pred

            self.corr = {
                el: torch.corrcoef(value.float().T)
                for el, value in self.true_vs_pred.items()
            }
            
    
            
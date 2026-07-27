import torch 
import numpy as np 
from dataclasses import dataclass
import numpy as np
from Bio.Seq import Seq
import json
from contextlib import nullcontext
from pathlib import Path

from downstream_tasks.expression_prediction.expression_dataset_final import ExpressionDataset

@dataclass
class ScoredSeq:
    seq: torch.FloatTensor
    dt: dict
    score: float
    method: str

class MaxCriterion: # should be minimized
    def __init__(self, targets: list[str], off_targets: list[str], maxes: dict[str, float]):
        self.maxes = maxes
        self.targets = targets
        self.off_targets = off_targets

    def __call__(self, pred: dict[str, float]):
        score = min(pred[ta] / self.maxes[ta] for ta in self.targets) - max(pred[ot] / self.maxes[ot] for ot in self.off_targets)
        return -score 

def _load_description_texts(selected_keys, description_source=None):
    if description_source is None:
        return {key: key for key in selected_keys}

    if isinstance(description_source, dict):
        missing = [key for key in selected_keys if key not in description_source]
        if missing:
            raise KeyError(f"Missing descriptions for selected keys: {missing}")
        return {key: description_source[key] for key in selected_keys}

    source_path = Path(description_source).expanduser()
    if source_path.is_dir():
        json_paths = {path.stem: path for path in source_path.rglob("*.json")}
        missing = [key for key in selected_keys if key not in json_paths]
        if missing:
            raise KeyError(
                f"Could not find JSON metadata files for selected keys: {missing}"
            )
        texts = {}
        for key in selected_keys:
            with open(json_paths[key], "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            texts[key] = ExpressionDataset.make_description_from_json(
                meta=meta,
                description_id=key,
                meta_path=str(json_paths[key]),
            )
        return texts

    if source_path.is_file() and source_path.suffix == ".json":
        with open(source_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if all(key in data and isinstance(data[key], str) for key in selected_keys):
            return {key: data[key] for key in selected_keys}
        if len(selected_keys) != 1:
            raise ValueError(
                "A single metadata JSON file can only be used with one selected key."
            )
        key = selected_keys[0]
        return {
            key: ExpressionDataset.make_description_from_json(
                meta=data,
                description_id=key,
                meta_path=str(source_path),
            )
        }

    raise ValueError(
        "description_source must be None, a dict, a metadata JSON file, "
        "or a directory with <selected_key>.json files."
    )


def _tokenize_descriptions(text_tokenizer, description_texts, selected_keys, max_length, device):
    encoded = text_tokenizer(
        [description_texts[key] for key in selected_keys],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "desc_input_ids": encoded["input_ids"].unsqueeze(0).to(device),
        "desc_attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
    }


def score_seq(
    model,
    tokenizer,
    seq,
    device,
    selected_keys,
    description_source=None,
    text_tokenizer=None,
    text_max_seq_len=510,
):
    if text_tokenizer is None:
        raise ValueError("text_tokenizer is required for ExpressionCounts scoring")

    with torch.inference_mode():
        X = encode(
            seq=seq,
            tokenizer=tokenizer,
            device=device,
            selected_keys=selected_keys,
            description_source=description_source,
            text_tokenizer=text_tokenizer,
            text_max_seq_len=text_max_seq_len,
        )
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if str(device).startswith("cuda")
            else nullcontext()
        )
        with autocast_context:
            output = model(**X)
        preds = output["logits"][:, 0, 0].detach().float().cpu()
        dt = {}
        for i, name in enumerate(selected_keys):
            dt[name] = preds[i].item()
        return dt 

def rev_seq(s: str) -> str:
    return str(Seq(s).reverse_complement())

# def encode(sq: str, tokenizer, device):
#     chunks = []
#     for i in range(0, len(sq), 128):
#         chunks.append(sq[i:i+128])
  
#     encoded_bins = tokenizer.batch_encode_plus(chunks, add_special_tokens=False, return_attention_mask=False,
#                                               return_token_type_ids=False)['input_ids']    
#     sample_token_ids = [tokenizer.cls_token_id]
#     for bin_token_ids in encoded_bins:
#         #if len(sample_token_ids) + len(bin_token_ids) + 1 < 512:
#         sample_token_ids.extend(bin_token_ids)
#         sample_token_ids.append(tokenizer.sep_token_id)
#     sample_token_ids = np.array(sample_token_ids)
#     token_type_ids = np.array([0] * len(sample_token_ids))
#     attention_mask = np.array([1] * len(sample_token_ids))
#     bins_mask = (sample_token_ids == tokenizer.sep_token_id).astype(bool)
#     X = {'input_ids': torch.from_numpy(sample_token_ids).unsqueeze(0).to(device),
#         'bins_mask': torch.from_numpy(bins_mask).unsqueeze(0).to(device),}
#     return X

def encode(
    seq: str,
    tokenizer,
    device,
    selected_keys,
    description_source=None,
    text_tokenizer=None,
    text_max_seq_len=510,
):
    encoded = tokenizer(
        seq,
        add_special_tokens=False,
        return_attention_mask=True,
        return_tensors="pt"  
    )
    n_keys = len(selected_keys)
    description_texts = _load_description_texts(selected_keys, description_source)
    desc_inputs = _tokenize_descriptions(
        text_tokenizer=text_tokenizer,
        description_texts=description_texts,
        selected_keys=selected_keys,
        max_length=text_max_seq_len,
        device=device,
    )

    #batch=1
    X = {
        "input_ids": encoded["input_ids"].repeat(n_keys, 1).to(device),
        "attention_mask": encoded["attention_mask"].repeat(n_keys, 1).to(device),
        "desc_input_ids": desc_inputs["desc_input_ids"],
        "desc_attention_mask": desc_inputs["desc_attention_mask"],
        "dataset_flag": torch.ones((1, n_keys), dtype=torch.bool, device=device),
    }
    return X

def get_score(model, 
              tokenizer, 
              seq: str,
              criterion, 
              device,  
              selected_keys,
              description_source=None,
              text_tokenizer=None,
              text_max_seq_len=510,
              label: str = "model"):
    with torch.inference_mode():
        dt = score_seq(
            model=model,
            tokenizer=tokenizer,
            seq=seq,
            device=device,
            selected_keys=selected_keys,
            description_source=description_source,
            text_tokenizer=text_tokenizer,
            text_max_seq_len=text_max_seq_len,
        )
        loss = criterion(dt)
    return ScoredSeq(seq=seq, dt=dt, score=loss, method=label)


def select_parents(population, T):
    scores = [p.score for p in population]
    ids = torch.multinomial(torch.softmax(-torch.FloatTensor(scores) / T, dim=0), 2)
    return population[ids[0]], population[ids[1]]

def get_default_mut_mask(length: int):
    return list(range(0, length * 3))

def single_point_mutate(seq, mut_mask):
    gen_pos =  torch.randint(low=0, high=len(mut_mask), size=(1, )).item()
    gen_pos = mut_mask.pop(gen_pos) # long op
    pos = gen_pos // 3
    alt_pos = gen_pos % 3
    alt = {'A', 'T', 'G', 'C'}
    alt.remove(seq[pos])
    seq = seq[:pos] + list(alt)[alt_pos] + seq[pos+1:]
    return seq


def get_probs(scores, T):
    return torch.softmax(-scores / T, dim=0)
    
def mix_sc(gen1, gen2, T):
    probs = get_probs(torch.FloatTensor([gen1.score, gen2.score]), T)
    
    take_first = torch.multinomial(probs, len(gen1.seq), replacement=True).bool().numpy()
    s = [gen1.seq[ind] if p else gen2.seq[ind] for ind, p in enumerate(take_first)]
    offspring = "".join(s)
    return offspring

def mix(gen1, gen2):
    probs = torch.FloatTensor([0.5, 0.5])
    take_first = torch.multinomial(probs, len(gen1.seq), replacement=True).bool().numpy()
    s = [gen1.seq[ind] if p else gen2.seq[ind] for ind, p in enumerate(take_first)]
    offspring = "".join(s)
    return offspring


def cross_sc(gen1, gen2, T, mean_length):
    probs = get_probs(torch.FloatTensor([gen1.score, gen2.score]), T)
    dists = torch.distributions.Poisson(rate=mean_length-1).sample(  (len(gen1.seq),) ) .long() + 1
    length = torch.cumsum(dists, 0)
    dists = dists[length <= len(gen1.seq)]
    dists[torch.randint(0, dists.shape[0], size=(1, ))] += len(gen1.seq) - length[dists.shape[0]-1]
    take_first = torch.multinomial(torch.FloatTensor(probs), len(gen1.seq), replacement=True).bool().numpy()
    
    s = [gen1.seq[ind] if p else gen2.seq[ind] for ind, p in enumerate(take_first)]
    offspring = "".join(s)
    return offspring

def cross(gen1, gen2, mean_length):
    probs = torch.FloatTensor([0.5, 0.5])
    dists = torch.distributions.Poisson(rate=mean_length-1).sample(  (len(gen1.seq),) ) .long() + 1
    length = torch.cumsum(dists, 0)
    dists = dists[length <= len(gen1.seq)]
    dists[torch.randint(0, dists.shape[0], size=(1, ))] += len(gen1.seq) - length[dists.shape[0]-1]
    take_first = torch.multinomial(torch.FloatTensor(probs), len(gen1.seq), replacement=True).bool().numpy()
    
    s = [gen1.seq[ind] if p else gen2.seq[ind] for ind, p in enumerate(take_first)]
    offspring = "".join(s)
    return offspring


def select_lowk(population, k):
    population.sort(key=lambda x: x.score)
    return population[:k]

def select_soft(population, k, T):
    scores = torch.FloatTensor([p.score for p in population])
    probs = get_probs(scores, T=T)
    poses = torch.multinomial(probs, num_samples=k).numpy().tolist()
    population = [population[i] for i in poses ]
    return population

def select_lowk_and_soft(population, k, T):
    if len(population) < 2 * k:
        return population
        
    population = sorted(population, key=lambda x: x.score)
    selected = population[:k] + select_soft(population[k:], k, T)

    return selected

def remove_duplicates(population):
    new_population = [population[0]]
    for ip in range(1, len(population)):
        for nip in range(len(new_population)):
            if population[ip].seq == new_population[nip].seq:
                break
        else:
            new_population.append(population[ip])
    return new_population

def select_lowk(population, k):
    population.sort(key=lambda x: x.score)
    return population[:k]

def select_soft(population, k, T):
    scores = torch.FloatTensor([p.score for p in population])
    probs = get_probs(scores, T=T)
    poses = torch.multinomial(probs, num_samples=k).numpy().tolist()
    population = [population[i] for i in poses ]
    return population

def select_lowk_and_soft(population, k, T):
    if len(population) < 2 * k:
        return population
        
    population = sorted(population, key=lambda x: x.score)
    selected = population[:k] + select_soft(population[k:], k, T)

    return selected

def remove_duplicates(population):
    new_population = [population[0]]
    for ip in range(1, len(population)):
        for nip in range(len(new_population)):
            if population[ip].seq == new_population[nip].seq:
                break
        else:
            new_population.append(population[ip])
    return new_population

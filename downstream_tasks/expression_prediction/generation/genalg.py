import torch 
import numpy as np 
from dataclasses import dataclass
import numpy as np
from Bio.Seq import Seq
import pickle

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

# def score_seq(model, tokenizer, seq, criterion, device, target_poses: dict[str, int]):
#     with torch.inference_mode():

#         X = encode(seq, tokenizer, device)

#         sc = model(**X)
#         sc = torch.cat(sc.logits_segm)
#         sc = torch.nn.functional.softplus(sc[:])

#         dt = {}
#         for name, p in target_poses.items():
#             dt[name] = sc[:, p].max(axis=0).values.detach().cpu().numpy().max()
#     return dt 

def score_seq(model, tokenizer, seq, device, selected_keys, text_data_path):
    with torch.inference_mode():
        X = encode(seq, tokenizer, device, selected_keys, text_data_path)
        output = model(**X)
        predictions_segm = [[el.detach().cpu() for el in s] for s in output['logits_segm']]
        rmt_labels_masks_segm = [[el.detach().cpu().to(torch.bool) for el in s] for s in output['labels_mask_reshaped']]
        preds = torch.stack(predictions_segm[-1])
        masks = torch.stack(rmt_labels_masks_segm[-1])
        p_segm = preds[:, 0, :].squeeze(-1)
        mask = masks[:, 0, :].squeeze(-1)
        p = torch.nn.functional.softplus(p_segm[mask])
        dt = {}
        for i, name in enumerate(selected_keys):
            dt[name]=p[i].item()
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

def load_description(selected_keys, text_data_path):
    with open(text_data_path, "rb") as f:
        desc_data = pickle.load(f)

    desc_vectors_list = []
    for key in selected_keys:
        if key not in desc_data:
            raise KeyError(f"Track ID '{key}' not found in desc_data")
        desc_vectors_list.append(desc_data[key])

    desc_vectors_np = np.stack(desc_vectors_list)
    desc_vectors = torch.from_numpy(desc_vectors_np)
    return desc_vectors

def encode(sq: str, tokenizer, device, selected_keys, text_data_path):
    desc_vectors = load_description(selected_keys, text_data_path)

    encoded = tokenizer(
        sq,
        add_special_tokens=False,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt"  
    )

    L = encoded["input_ids"].shape[1]
    N_keys = desc_vectors.shape[0]

    labels_mask = torch.zeros((1, L, N_keys), dtype=torch.bool, device=device)
    tpm = torch.ones((1, N_keys), dtype=torch.float32, device=device)
    labels = torch.zeros((1, L, N_keys), dtype=torch.float32, device=device) 

    #batch=1
    X = {
        'input_ids':      encoded["input_ids"].to(device),       # [1, L]
        'attention_mask': encoded["attention_mask"].to(device),  # [1, L]
        'token_type_ids': encoded["token_type_ids"].to(device),  # [1, L]
        'labels_mask':    labels_mask,                    # [1, L, N_keys]
        'labels':            labels,                      # [1, L, N_keys]
        'desc_vectors':   desc_vectors.unsqueeze(0).to(device),  # [1, N_keys, D_desc]
        'tpm':            tpm                         # [1, N_keys]
    }
    return X

def get_score(model, 
              tokenizer, 
              seq: str,
              criterion, 
              device,  
              selected_keys, text_data_path,
              label: str):
    with torch.inference_mode():
        dt = score_seq(model=model, tokenizer=tokenizer, seq=seq, device=device, selected_keys = selected_keys, text_data_path = text_data_path)
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


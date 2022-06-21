import os
import Bio
from Bio import SeqIO
import numpy as np
import pandas as pd

print('Please, input fasta filename: ')
path = input()
ind = path.find('len')
name = path.split('.')[0]
length = name[ind:]

count = 0
sequences = [] # Here we are setting up an array to save our sequences for the next step
for seq_record in SeqIO.parse(path, "fasta"):
    sequences.append(seq_record.seq)
    count = count + 1
if len(sequences[0]) > 16000:   
    positive_seqs = [str(sequence[:16000]) for sequence in sequences]
else:
    positive_seqs = [str(sequence) for sequence in sequences]

def sample_negative(sequence):
    np.random.seed(42)
    assert len(sequence)%20==0, 'Sequence length should be divisible by 20. E.g. 300 = epdnew from -249 to 50'
    n = len(sequence)
    step = n//20
    subs = [ sequence[i:i+step] for i in range(0, n, step) ]
    selected_inds = list(np.random.choice(20, 12, replace=False))
    selected = [subs[i] for i in selected_inds]
    not_selected_inds = list(set(range(20)).difference(selected_inds))
    not_selected = [subs[i] for i in not_selected_inds]
    
    new_s = ''
    np.random.shuffle(selected)
    for i in range(20):
        if i in selected_inds:
            new_s += selected.pop(0)
        else:
            new_s += not_selected.pop(0)
    return new_s

negative_seqs = []
for s in positive_seqs:
    negative_seqs.append(sample_negative(s))

# Generate dataset
l = len(positive_seqs)
all_seqs = positive_seqs.copy()
all_seqs.extend(negative_seqs)
len(all_seqs)

df = pd.DataFrame.from_dict({'sequence' : all_seqs, 'promoter_presence' : [1]*l + [0]*l})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(f'hg38_{length}_promoters_dataset.csv', index=False)

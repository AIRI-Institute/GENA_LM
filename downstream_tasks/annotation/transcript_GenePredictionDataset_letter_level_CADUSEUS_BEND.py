#!/usr/bin/env python
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch


class AnnotationDataset(Dataset):
    def __init__(
        self,
        targets_file,
        max_atcg_seq_len,
        tokenizer,
        tokenizer_caduseus,
        tmp_valid_len = None,
        shuffle_starts = False
    ):
        self.data = h5py.File(targets_file, "r")
        self.max_atcg_seq_len = max_atcg_seq_len
        self.tokenizer = tokenizer
        self.tokenizer_caduseus = tokenizer_caduseus
        self.tmp_valid_len = tmp_valid_len
        self.shuffle_starts = shuffle_starts

        self.samples = len(list(self.data.keys()))

    def shuffle_starts_func(self, labels, token_atcg):
        # print('YES')
        chooser = np.random.rand()
        # print(chooser)
        if chooser > 0.5:
            ir_pos_idx = np.arange(labels.shape[0])
            counter = 0
            while True:
                counter += 1
                selected_start = np.random.choice(ir_pos_idx)
                if len(labels) - selected_start > 512: # some threshold
                    break
                if counter > 100:
                    return labels, token_atcg

            # print(selected_start, len(labels))
            token_atcg = token_atcg[selected_start:]
            labels = labels[selected_start:, :]
        return labels, token_atcg

    def __len__(self):
        return self.tmp_valid_len if self.tmp_valid_len is not None else self.samples

    def __getitem__(self, idx):
        assert idx < self.samples
        
        if self.tmp_valid_len is None:
            sample_name = "transcript_" + str(idx)
        else:
            sample_name = "transcript_" + str(int(np.random.randint(0, self.samples)))

        labels = np.array(
            self.data[sample_name]["labels"]
        ).astype(int)[:]

        new_labels = np.zeros((len(labels), 9))
        new_labels[np.arange(len(labels)), labels] = 1
        labels = new_labels[:, :]
 
        token_atcg = np.array(
            self.data[sample_name]["token_atcg"]
        )[:]

        assert len(token_atcg) == len(labels)
        assert token_atcg[-1] != 2
        assert token_atcg[0] != 1

        if self.shuffle_starts:
            labels, token_atcg = self.shuffle_starts_func(labels, token_atcg)

        
        assert len(token_atcg) == len(labels)
        assert token_atcg[-1] != 2
        assert token_atcg[0] != 1

        atcg_seq = ''.join(list(self.tokenizer.decode(token_atcg)))
        # print(atcg_seq)
        labels = labels[:self.max_atcg_seq_len-1, :]
        atcg_seq = atcg_seq[:self.max_atcg_seq_len-1]
        token_atcg = self.tokenizer_caduseus(atcg_seq, return_tensors='np')['input_ids'].squeeze()
        # print(token_atcg)
        # assert False

        # print(token_atcg)
        assert 1 in token_atcg

        if labels.shape[1] < self.max_atcg_seq_len:
            labels = np.pad(labels, pad_width=((self.max_atcg_seq_len - labels.shape[0] - 1, 1), (0, 0)), mode='constant', constant_values=-100)

        if len(token_atcg) < self.max_atcg_seq_len:
            token_atcg = np.pad(token_atcg, pad_width=((self.max_atcg_seq_len - token_atcg.shape[0], 0)), mode='constant', constant_values=4)

        assert len(token_atcg) == len(labels)
        assert len(token_atcg) == self.max_atcg_seq_len

        letter_level_mask = (token_atcg != 4) & (token_atcg != 1)

        assert len(token_atcg[letter_level_mask]) == len(atcg_seq)
        assert len(token_atcg[letter_level_mask]) == len(labels[letter_level_mask, :])
        assert -100 not in labels[letter_level_mask, :]
        assert 1 in token_atcg
        assert 1 not in token_atcg[letter_level_mask]
        
        return {
            "input_ids": token_atcg.squeeze(),
            "letter_level_labels": labels.squeeze().astype(float),
            "letter_level_labels_mask": letter_level_mask.squeeze()
        }

    def close(self):
        self.data.close()

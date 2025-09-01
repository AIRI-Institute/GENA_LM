#!/usr/bin/env python
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch


class AnnotationDataset(Dataset):
    def __init__(
        self,
        targets_file,
        max_seq_len,
        max_atcg_seq_len,
        tokenizer,
        tmp_valid_len = None,
        shuffle_starts = False
    ):
        self.data = h5py.File(targets_file, "r")
        self.max_seq_len = max_seq_len
        self.max_atcg_seq_len = max_atcg_seq_len
        self.tokenizer = tokenizer
        self.tmp_valid_len = tmp_valid_len
        self.shuffle_starts = shuffle_starts

        self.samples = len(list(self.data.keys()))

    def __len__(self):
        return self.tmp_valid_len if self.tmp_valid_len is not None else self.samples

    def shuffle_starts_func(self, input_ids, labels, token_atcg):
        chooser = np.random.rand()
        # print(chooser)
        if chooser > 0.5:
            ir_pos_idx = np.arange(labels.shape[1])
            counter = 0
            while True:
                counter += 1
                selected_start = np.random.choice(ir_pos_idx)
                if len(labels) - selected_start > 512: # some threshold
                    break
                if counter > 100:
                    return input_ids, labels, token_atcg

            # print(selected_start, len(labels))
            token_atcg = token_atcg[:, selected_start:]
            input_ids = self.tokenizer([self.tokenizer.decode(token_atcg[0, :])], return_tensors='np')['input_ids']
            labels = labels[:, selected_start:, :]
        return input_ids, labels, token_atcg

    def __getitem__(self, idx):
        assert idx < self.samples
        
        if self.tmp_valid_len is None:
            sample_name = "transcript_" + str(idx)
        else:
            sample_name = "transcript_" + str(int(np.random.randint(0, self.samples)))

        input_ids = np.array(
            self.data[sample_name]["input_ids"]
        )[None, :]
        # token_type_ids = np.array(
        #     self.data[sample_name]["token_type_ids"]
        # )
        # attention_mask = np.array(
        #     self.data[sample_name]["attention_mask"]
        # )
        labels = np.array(
            self.data[sample_name]["labels_atcg"]
        )[None, :, :]#[None, :, :]#[None, 1:-1, :]
        # new_labels = np.zeros((len(labels), 24))
        # new_labels[np.arange(len(labels)), labels] = 1
        # labels = new_labels[None, :, :]
        token_atcg = np.array(
            self.data[sample_name]["token_atcg"]
        )[None, :]#[None, :]#[None, 1:-1]
        # print(token_atcg)
        # print(input_ids.shape, labels.shape, token_atcg.shape)
        
        assert len(token_atcg) == len(labels)
        assert input_ids[0, -1] == 2
        assert input_ids[0, 0] == 1
        assert token_atcg[0, -1] != 2
        assert token_atcg[0, 0] != 1

        if self.shuffle_starts:
            input_ids, labels, token_atcg = self.shuffle_starts_func(input_ids, labels, token_atcg)

        
        assert len(token_atcg) == len(labels)
        assert input_ids[0, -1] == 2
        assert input_ids[0, 0] == 1
        assert token_atcg[0, -1] != 2
        assert token_atcg[0, 0] != 1

        initial_len_input_ids = input_ids.shape[1]
        # print('OLOLOLOL', input_ids.shape)
        # print('##########################', len(list(input_ids[0, :])))
        if initial_len_input_ids < self.max_seq_len:
            input_ids = np.array(list(input_ids[0, :]) + [3] * (self.max_seq_len - input_ids.shape[1])).astype(int).reshape((1, -1))
        else:
            input_ids = np.array(list(input_ids[0, :])[:self.max_seq_len-1] + [2]).astype(int).reshape((1, -1))
        # print('IIIIIIIIIIIIIIII', input_ids.shape)
        
        token_type_ids = np.zeros((1, self.max_seq_len))
        attention_mask = (input_ids != 3).astype(int)
        
        atcg_seq = ''
        token_repeater_numbers = []
        meaningful_tokens_only = input_ids[0][input_ids[0] > 5]
        for t in meaningful_tokens_only:
            atcg_seq_token = self.tokenizer.convert_ids_to_tokens(int(t))
            token_repeater_numbers.append(len(atcg_seq_token))
            atcg_seq += atcg_seq_token
            
        # print('LEN FULL TRANSCRIPT', len(atcg_seq), len(token_repeater_numbers), sum(token_repeater_numbers))
        assert len(atcg_seq) == sum(token_repeater_numbers)
        
        token_repeater = []
        for n, i in enumerate(token_repeater_numbers):
            # print(i)
            for j in range(i):
                token_repeater.append(n)
                
        # print('LEN TOKEN REPEATER', input_ids.shape, labels.shape, token_atcg.shape, len(token_repeater))
        
        assert len(token_repeater) <= labels.shape[1]
        
        labels = labels[:, :len(token_repeater), :]
        token_atcg = token_atcg[:, :len(token_repeater)]
        
        if len(token_repeater) < self.max_atcg_seq_len:
            labels = np.concatenate((labels, np.full((1, self.max_atcg_seq_len - len(token_repeater), 5), -100)), axis=1)
            token_atcg = np.concatenate((token_atcg, np.full((1, self.max_atcg_seq_len - len(token_repeater)), -100)), axis=1)
            token_repeater = token_repeater + [-100] * (self.max_atcg_seq_len - len(token_repeater))
        else:
            token_repeater = token_repeater[:self.max_atcg_seq_len]
            labels = labels[:, :self.max_atcg_seq_len, :]
            token_atcg = token_atcg[:, :self.max_atcg_seq_len]
        token_repeater = np.array(token_repeater).reshape((1, -1)).astype(int)
        
        letter_level_mask = token_repeater != -100

        # print('FINAL', input_ids.shape, token_type_ids.shape, attention_mask.shape, labels.shape, token_atcg.shape)
        
        return {
            "input_ids": input_ids.squeeze(),
            "token_type_ids": token_type_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "labels": torch.randint(0, 5, (input_ids.shape[1], 5)), # change it in future
            "labels_ohe": torch.randint(0, 5, (input_ids.shape[1], 5)), # change it in future
            "labels_mask": (input_ids.squeeze() > 5).astype(int),
            "letter_level_tokens": token_atcg.squeeze(), 
            "letter_level_labels": labels.squeeze(),
            "letter_level_labels_mask": letter_level_mask.squeeze(),
            "embedding_repeater": token_repeater.squeeze(),
            "letter_level_attention_mask" : letter_level_mask.astype(int).squeeze(), # np.ones(self.max_atcg_seq_len).astype(int),
            "letter_level_token_types_ids": np.zeros(self.max_atcg_seq_len).astype(int)
        }

    def close(self):
        self.data.close()

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
        tmp_valid_len = None
    ):
        self.data = h5py.File(targets_file, "r")
        self.max_seq_len = max_seq_len
        self.max_atcg_seq_len = max_atcg_seq_len
        self.tokenizer = tokenizer
        self.tmp_valid_len = tmp_valid_len

        self.samples = len(list(self.data.keys()))

    def __len__(self):
        return self.tmp_valid_len if self.tmp_valid_len is not None else self.samples

    def process_data(self, input_ids, labels, token_atcg):

        assert input_ids[0, -1] == 2

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
                
        # print('LEN TOKEN REPEATER', len(token_repeater))
        
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

        return input_ids, token_type_ids, attention_mask, token_atcg, labels, letter_level_mask, token_repeater

    def __getitem__(self, idx):
        assert idx < self.samples
        
        if self.tmp_valid_len is None:
            sample_name = "transcript_" + str(idx)
        else:
            sample_name = "transcript_" + str(int(np.random.randint(0, self.samples)))

        # input_ids_forward = np.array(
        #     self.data[sample_name]["input_ids_forward"]
        # )[None, :]
        input_ids_backward = np.array(
            self.data[sample_name]["input_ids_reverse"]
        )[None, :]
        # token_type_ids = np.array(
        #     self.data[sample_name]["token_type_ids"]
        # )
        # attention_mask = np.array(
        #     self.data[sample_name]["attention_mask"]
        # )
        labels = np.array(
            self.data[sample_name]["labels_atcg_forward"]
        )[None, :, :]
        token_atcg = np.array(
            self.data[sample_name]["token_atcg_forward"]
        )[None, :]
        # print(input_ids_backward.shape, labels.shape, token_atcg.shape)
        # print(token_atcg)
        # print(input_ids.shape, token_type_ids.shape, attention_mask.shape, labels.shape, token_atcg.shape)
        
        input_ids_backward, token_type_ids_backward, attention_mask_backward, _, _, _, token_repeater_backward = self.process_data(input_ids_backward, labels, token_atcg)

        input_ids_forward = np.array(
            self.data[sample_name]["input_ids_forward"]
        )[None, :]
        # input_ids_backward = np.array(
        #     self.data[sample_name]["input_ids_reverse"]
        # )[None, :]
        labels = np.array(
            self.data[sample_name]["labels_atcg_forward"]
        )[None, :, :]
        token_atcg = np.array(
            self.data[sample_name]["token_atcg_forward"]
        )[None, :]

        input_ids_forward, token_type_ids_forward, attention_mask_forward, token_atcg, labels, letter_level_mask, token_repeater_forward = self.process_data(input_ids_forward, labels, token_atcg)

        # print('FINAL', input_ids.shape, token_type_ids.shape, attention_mask.shape, labels.shape, token_atcg.shape)
        # print(input_ids_forward.shape, input_ids_backward.shape)
        assert input_ids_forward.shape[1] == input_ids_backward.shape[1]
        
        return {
            "input_ids_forward": input_ids_forward.squeeze().astype(int),
            "token_type_ids_forward": token_type_ids_forward.squeeze().astype(int),
            "attention_mask_forward": attention_mask_forward.squeeze().astype(int),
            "input_ids_backward": input_ids_backward.squeeze().astype(int),
            "token_type_ids_backward": token_type_ids_backward.squeeze().astype(int),
            "attention_mask_backward": attention_mask_backward.squeeze().astype(int),
            "labels": torch.randint(0, 5, (input_ids_forward.shape[1], 5)), # change it in future
            "labels_ohe": torch.randint(0, 5, (input_ids_forward.shape[1], 5)), # change it in future
            "labels_mask_forward": (input_ids_forward.squeeze() > 5).astype(int),
            "labels_mask_backward": (input_ids_backward.squeeze() > 5).astype(int),
            "letter_level_tokens": token_atcg.squeeze(), 
            "letter_level_labels": labels.squeeze(),
            "letter_level_labels_mask": letter_level_mask.squeeze(),
            "embedding_repeater_forward": token_repeater_forward.squeeze().astype(int),
            "embedding_repeater_backward": token_repeater_backward.squeeze().astype(int),
            "letter_level_attention_mask" : letter_level_mask.astype(int).squeeze(), # np.ones(self.max_atcg_seq_len).astype(int),
            "letter_level_token_types_ids": np.zeros(self.max_atcg_seq_len).astype(int)
        }

    def close(self):
        self.data.close()

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
        tmp_valid_len,
        use_shortener
    ):
        self.data = h5py.File(targets_file, "r")
        self.max_seq_len = max_seq_len
        
        self.tmp_valid_len = tmp_valid_len

        self.samples = len(list(self.data.keys()))
        self.use_shortener = use_shortener

    def __len__(self):
        return self.samples if self.tmp_valid_len is None else self.tmp_valid_len

    def __getitem__(self, idx):
        assert idx < self.samples

        sample_name = "transcript_" + str(idx)

        # print(self.data[sample_name].keys())
        
        # print(np.array(self.data["transcript_" + str(1272)]["input_ids"]))

        input_ids = np.array(
            self.data[sample_name]["input_ids_forward"]
        )
        # token_type_ids = np.array(
        #     self.data[sample_name]["token_type_ids"]
        # )
        # attention_mask = np.array(
        #     self.data[sample_name]["attention_mask"]
        # )
        labels = np.array(
            self.data[sample_name]["labels_forward"]
        )
        # labels = labels.T
        labels = labels.astype(np.float32)
        # print(input_ids.shape, labels.shape)
        
        if self.use_shortener:
            if len(input_ids) > 1024:
                random_pos = np.random.randint(1024, len(input_ids))
                input_ids = np.concatenate((input_ids[:random_pos], input_ids[-1:]))
                labels = np.concatenate((labels[:random_pos, :], labels[-1:, :]), axis=0)
        
        assert input_ids[-1] == 2
        assert input_ids[0] == 1
        # assert 1 in labels[:, 1]
        assert -100 in labels[0, :]
        assert -100 in labels[-1, :]
        # input_ids = list(input_ids[:-1][input_ids[:-1] != 3])
        # input_ids = input_ids + [2]
        initial_len_input_ids = len(input_ids)
        if initial_len_input_ids < self.max_seq_len:
            input_ids = np.array(list(input_ids) + [3] * (self.max_seq_len - len(input_ids))).astype(int)
        else:
            input_ids = np.array(list(input_ids)[:self.max_seq_len-1] + [2]).astype(int)
        # print(len(input_ids))
        
        token_type_ids = np.zeros(self.max_seq_len)
        
        attention_mask = (input_ids != 3).astype(int)
        labels_mask = np.array([0 if i<=5 else 1 for i in input_ids]).astype(float)
        
        if initial_len_input_ids < self.max_seq_len:
            labels = np.array([[-100, -100, -100, -100, -100, -100] if input_ids[idx] <=5 else i for idx, i in enumerate(labels)] + [[-100, -100, -100, -100, -100, -100]] * (self.max_seq_len - len(labels)))
        else:
            labels = np.array([[-100, -100, -100, -100, -100, -100] if input_ids[idx] <=5 else i for idx, i in enumerate(labels[:self.max_seq_len-1, :])] + [[-100, -100, -100, -100, -100, -100]])
            
        assert (
            len(input_ids) == len(token_type_ids)
            and len(token_type_ids) == len(attention_mask)
        )
        
        # n_labels = 6
        # labels_ohe = np.zeros((len(labels), n_labels))
        # for label in range(n_labels):
        #     labels_ohe[(labels == label).max(axis=-1), label] = 1.0

            
        # print('inputs', input_ids)
        # print('tti', token_type_ids)
        # print('am', attention_mask)
        # print('lab', labels)
        # print('labm', labels_mask)
        
        
        assert len(input_ids) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(labels) == self.max_seq_len
        # assert len(labels_ohe) == self.max_seq_len
        assert len(labels_mask) == self.max_seq_len
                                                                                                               
                                                                                                               
        # set mask to 0.0 for tokens with no labels, these examples should not be used for loss computation
        # labels_mask = np.ones(len(labels))
        # labels_mask[labels_ohe.sum(axis=-1) == 0.0] = 0.0
        
        # print('AAAA', np.sum(input_ids == 2))
        # print('ids', input_ids[-3:])
        # print('mask', labels_mask[-3:])
        # print('labels', labels[-3:])
        # print('am', attention_mask[-3:])
        # print('tti', token_type_ids[-3:])

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "labels_ohe": labels,
            "labels_mask": labels_mask # labels.min(axis=1) != -100
        }

    def close(self):
        self.data.close()

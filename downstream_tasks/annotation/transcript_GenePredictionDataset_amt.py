#!/usr/bin/env python
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import math


class AnnotationDataset(Dataset):
    def __init__(
        self,
        targets_file,
        max_seq_len,
        tmp_valid_len,
        use_shortener,
        segment_size
    ):
        self.data = h5py.File(targets_file, "r")
        self.max_seq_len = max_seq_len
        
        self.tmp_valid_len = tmp_valid_len

        self.samples = len(list(self.data.keys()))
        self.use_shortener = use_shortener
        self.segment_size = segment_size

    def __len__(self):
        return self.samples if self.tmp_valid_len is None else self.tmp_valid_len

    def insert_special_tokens(self, input_ids, labels, segment_size):
        """
        Process 1D input_ids and 2D labels by:
          1. Taking the tokens before the first occurrence of 3,
             splitting them into chunks of (segment_size-2) tokens,
             and wrapping each chunk with 1 at the beginning and 2 at the end.
          2. Leaving the tokens starting with the first 3 untouched.
        
        For any token that is 1, 2, or 3 in the output, the corresponding label is a
        vector (of the same shape as one row of labels) filled with -100.
        For other tokens, the label is copied from the original labels.
        
        Args:
          input_ids (np.ndarray): 1D numpy array of token ids.
          labels (np.ndarray): 2D numpy array of labels. (input_ids.shape[0] == labels.shape[0])
          segment_size (int): The desired segment size. (Each segment will actually have length=segment_size 
                                after inserting the special tokens 1 and 2.)
        
        Returns:
          new_input_ids, new_labels: The modified arrays.
        """
        # Sanity check:
        assert input_ids.shape[0] == labels.shape[0], "input_ids and labels must have the same number of rows."
        
        # Define our special tokens:
        CLS = 1
        SEP = 2
        SPECIAL = 3  # When we encounter a token 3 in input_ids, we stop grouping.
        
        group_size = segment_size - 2  # we will insert 2 extra tokens (CLS and SEP) per segment
    
        # Find the first occurrence of SPECIAL (i.e. token 3) in input_ids.
        special_idx = np.where(input_ids == SPECIAL)[0]
        if special_idx.size > 0:
            first_special = special_idx[0]
        else:
            first_special = input_ids.shape[0]
        
        # Tokens before the first SPECIAL will be grouped and wrapped.
        prefix_ids = input_ids[:first_special]
        prefix_labels = labels[:first_special]
        
        new_ids_parts = []
        new_labels_parts = []
        
        # Process the prefix in chunks of group_size.
        for start in range(0, prefix_ids.shape[0], group_size):
            chunk_ids = prefix_ids[start:start+group_size]
            chunk_labels = prefix_labels[start:start+group_size]
            
            # Create new segment: [CLS] + chunk + [SEP]
            seg_ids = np.concatenate(([CLS], chunk_ids, [SEP]))
            # For the inserted tokens, we create a row of -100 with the same number of columns as labels.
            special_label = np.full((1, labels.shape[1]), -100, dtype=labels.dtype)
            seg_labels = np.concatenate((special_label, chunk_labels, special_label), axis=0)
            
            new_ids_parts.append(seg_ids)
            new_labels_parts.append(seg_labels)
        
        # Now, process the suffix: all tokens starting from the first SPECIAL.
        # They are left as they are (no wrapping) but we force their labels to be -100.
        if first_special < input_ids.shape[0]:
            suffix_ids = input_ids[first_special:]
            suffix_labels = np.full((suffix_ids.shape[0], labels.shape[1]), -100, dtype=labels.dtype)
            new_ids_parts.append(suffix_ids)
            new_labels_parts.append(suffix_labels)
        
        # Concatenate everything.
        new_input_ids = np.concatenate(new_ids_parts)
        new_labels = np.concatenate(new_labels_parts, axis=0)
        
        return new_input_ids, new_labels

    def __getitem__(self, idx):
        assert idx < self.samples

        sample_name = "transcript_" + str(idx)

        # print(self.data[sample_name].keys())
        
        # print(np.array(self.data["transcript_" + str(1272)]["input_ids"]))

        input_ids = np.array(
            self.data[sample_name]["input_ids_forward"]
        )[1:-1]
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
        labels = labels.astype(np.float32)[1:-1]
        # print(input_ids.shape, labels.shape)
        
        if self.use_shortener:
            if len(input_ids) > 1024:
                random_pos = np.random.randint(1024, len(input_ids))
                input_ids = input_ids[:random_pos]
                labels = labels[:random_pos, :]
        
        assert input_ids[-1] != 2
        assert input_ids[0] != 1
        # assert 1 in labels[:, 1]
        assert -100 not in labels[0, :]
        assert -100 not in labels[-1, :]
        # input_ids = list(input_ids[:-1][input_ids[:-1] != 3])
        # input_ids = input_ids + [2]
        initial_len_input_ids = len(input_ids)
        if initial_len_input_ids < self.max_seq_len: # MEANING TOKENS, namesly, WITHOUT SEP and CLS
            input_ids = np.array(list(input_ids) + [3] * (self.max_seq_len - len(input_ids))).astype(int)
        else:
            input_ids = input_ids[:self.max_seq_len]
        # print(len(input_ids))
        
        
        
        
        
        if initial_len_input_ids < self.max_seq_len:
            labels = np.array([[-100, -100, -100, -100, -100, -100] if input_ids[idx] <=5 else i for idx, i in enumerate(labels)] + [[-100, -100, -100, -100, -100, -100]] * (self.max_seq_len - len(labels)))
        else:
            labels = np.array([[-100, -100, -100, -100, -100, -100] if input_ids[idx] <=5 else i for idx, i in enumerate(labels[:self.max_seq_len, :])])

        assert len(input_ids) == self.max_seq_len
        assert labels.shape[0] == self.max_seq_len

        input_ids, labels = self.insert_special_tokens(input_ids, labels, self.segment_size)

        max_elongation = math.ceil(self.max_seq_len / (self.segment_size - 2)) * 2 + self.max_seq_len
        input_ids = np.array(list(input_ids) + [3] * (max_elongation - len(input_ids))).astype(int)
        labels = np.array([[-100, -100, -100, -100, -100, -100] if input_ids[idx] <=5 else i for idx, i in enumerate(labels)] + [[-100, -100, -100, -100, -100, -100]] * (max_elongation - len(labels)))

        # print(initial_len_input_ids, input_ids.shape, labels.shape)

        token_type_ids = np.zeros(len(input_ids))
        attention_mask = (input_ids != 3).astype(int)
        labels_mask = np.array([0 if i<=5 else 1 for i in input_ids]).astype(float)


        
        assert (
            len(input_ids) == len(token_type_ids)
            and len(token_type_ids) == len(attention_mask)
        )

        assert input_ids[-1] == 2 or input_ids[-1] == 3
        assert input_ids[0] == 1
        # assert 1 in labels[:, 1]
        assert -100 in labels[0, :]
        
        # n_labels = 6
        # labels_ohe = np.zeros((len(labels), n_labels))
        # for label in range(n_labels):
        #     labels_ohe[(labels == label).max(axis=-1), label] = 1.0

            
        # print('inputs', input_ids)
        # print('tti', token_type_ids)
        # print('am', attention_mask)
        # print('lab', labels)
        # print('labm', labels_mask)
        
        
        assert len(input_ids) == max_elongation
        assert len(token_type_ids) == max_elongation
        assert len(attention_mask) == max_elongation
        assert len(labels) == max_elongation
        assert len(labels_mask) == max_elongation
                                                                                                               
                                                                                                               
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
            "labels_mask": labels_mask
        }

    def close(self):
        self.data.close()

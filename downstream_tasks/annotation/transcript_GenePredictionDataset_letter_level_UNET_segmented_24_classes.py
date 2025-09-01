# #!/usr/bin/env python
# import h5py
# import numpy as np
# from torch.utils.data import Dataset
# import torch
# import time


# class AnnotationDataset(Dataset):
#     def __init__(
#         self,
#         targets_file,
#         max_seq_len,
#         max_atcg_seq_len,
#         tokenizer,
#         tmp_valid_len=None,
#         shuffle_starts=False
#     ):
#         # ──────────────────────────────────────────────────────────────────
#         # Load the entire HDF5 file into RAM
#         # ──────────────────────────────────────────────────────────────────
#         self.samples_data = {}
#         with h5py.File(targets_file, "r") as h5f:
#             for k in h5f.keys():
#                 self.samples_data[k] = {d: h5f[k][d][()] for d in h5f[k].keys()}

#         # Decide dataset type exactly once
#         first_key = next(iter(self.samples_data))
#         self.is_sample_dataset = first_key.startswith("sample_")

#         self.max_seq_len        = max_seq_len
#         self.max_atcg_seq_len   = max_atcg_seq_len
#         self.tokenizer          = tokenizer
#         self.tmp_valid_len      = tmp_valid_len
#         self.shuffle_starts     = shuffle_starts
#         self.eye24 = np.eye(24)

#         self.keys    = list(self.samples_data.keys())
#         self.samples = len(self.keys)

#     # ------------------------------------------------------------------
#     def __len__(self):
#         return self.tmp_valid_len if self.tmp_valid_len is not None else self.samples

#     def shuffle_starts_func(self, input_ids, labels, token_atcg):
#         chooser = np.random.rand()
#         # print(chooser)
#         if chooser > -1:
#             ir_pos_idx = np.arange(labels.shape[1])
#             counter = 0
#             while True:
#                 counter += 1
#                 selected_start = np.random.choice(ir_pos_idx)
#                 if len(labels) - selected_start > 512: # some threshold
#                     break
#                 if counter > 100:
#                     return input_ids, labels, token_atcg

#             # print(selected_start, len(labels))
#             token_atcg = token_atcg[:, selected_start:]
#             input_ids = self.tokenizer([self.tokenizer.decode(token_atcg[0, :])], return_tensors='np')['input_ids']
#             labels = labels[:, selected_start:, :]
#         return input_ids, labels, token_atcg

#     # ------------------------------------------------------------------
#     def __getitem__(self, idx):
#         # st = time.time()
#         assert idx < self.samples
#         sample_name = self.keys[idx] if self.tmp_valid_len is None \
#                       else self.keys[np.random.randint(0, self.samples)]
#         # print('idx', time.time() - st)

#         # st = time.time()
#         sample    = self.samples_data[sample_name]
#         is_sample = self.is_sample_dataset      # == sample_name.startswith("sample_")
#         # print('sampling', time.time() - st)

#         # st = time.time()
#         token_atcg_full = sample["token_atcg"][None, :]
#         labels_full     = sample["labels"]   
#         overlap_mask    = sample.get("overlaps", None)
#         # print('getting', time.time() - st)

#         # ──────────────────────────────────────────────────────────────
#         # NEW / CHANGED:  strict mask handling for *sample_* datasets
#         # ──────────────────────────────────────────────────────────────
#         if is_sample:
#             # st = time.time()
#             total_len = token_atcg_full.shape[1]
#             # if overlap_mask is not None:
#             #     overlap_mask = overlap_mask.astype(bool)
#             # print('bool operation', time.time() - st)

#             found = False
#             attempts = 0
#             while not found:

#                 # st = time.time()
#                 attempts += 1
#                 # pick a random window of max_atcg_seq_len
#                 start = np.random.randint(0, total_len - self.max_atcg_seq_len + 1)
#                 end   = start + self.max_atcg_seq_len
#                 seg   = overlap_mask[start:end].astype(bool)
#                 # print('mask sampling', time.time() - st)

#                 # identify contiguous runs of 0s and 1s
#                 # st = time.time()
#                 zero_idx = np.where(~seg)[0]
#                 if zero_idx.size == 0:
#                     zero_runs = 0
#                 else:
#                     z_splits  = np.split(zero_idx, np.where(np.diff(zero_idx) != 1)[0] + 1)
#                     zero_runs = len([r for r in z_splits if len(r) > 0])
#                 # print('diff, where, split', time.time() - st)

#                 if zero_runs > 1:                 # ─── NEW / CHANGED ───► rule‑2
#                     continue                      # resample immediately

#                 # contiguous 1‑runs
#                 # st = time.time()
#                 ones_idx = np.where(seg)[0]
#                 if ones_idx.size == 0:            # no allowed bases at all
#                     continue

#                 o_splits = np.split(ones_idx, np.where(np.diff(ones_idx) != 1)[0] + 1)
#                 good_runs = [r for r in o_splits if len(r) >= 512]   # rule‑4

#                 if len(good_runs) != 1:           # must be exactly ONE good run
#                     continue

#                 # print('diff, where, split 2', time.time() - st)

#                 # st = time.time()
#                 chosen_run = good_runs[0]
#                 run_start, run_end = chosen_run[0], chosen_run[-1] + 1  # inclusive→exclusive
#                 # drop disallowed edges
#                 # print(int(start), int(run_start), int(start), int(run_end))
#                 token_atcg = token_atcg_full[:, int(start) + int(run_start) : int(start) + int(run_end)]
#                 labels     = labels_full[int(start) + int(run_start) : int(start) + int(run_end)]
#                 labels = self.eye24[labels]       # shape (L,24), view-based gather
#                 labels   = labels[None, :, :]
#                 assert labels.shape[-1] == 24
#                 # print('slicing', time.time() - st)

#                 found = True
#                 if attempts > 1000:              # emergency fallback
#                     raise "How that happened?"
#                     # token_atcg = token_atcg_full[:, :self.max_atcg_seq_len]
#                     # labels     = labels_full[:,  :self.max_atcg_seq_len, :]
#                     # break

            

#             # decode only the kept slice
#             # start = time.time()
#             atcg_seq_slice = self.tokenizer.decode(token_atcg[0, :])
#             input_ids = self.tokenizer(
#                 atcg_seq_slice,
#                 return_tensors='np'
#             )['input_ids']
#             # print('decoding', time.time() - start)

#         else:
#             # old dataset: everything pre‑computed
#             input_ids  = sample["input_ids"][None, :]
#             token_atcg = token_atcg_full
#             labels     = labels_full
#             labels = self.eye24[labels_full]       # shape (L,24), view-based gather
#             labels   = labels[None, :, :]

#             # -------- optional shuffle for BOTH dataset types ---------------
#             if self.shuffle_starts:
#                 input_ids, labels, token_atcg = self.shuffle_starts_func(
#                     input_ids, labels, token_atcg
#                 )

#         # -------------------------- asserts -----------------------------
#         # print(len(token_atcg), len(labels))
#         assert token_atcg.shape[1] == labels.shape[1]
#         assert input_ids[0, -1] == 2
#         assert input_ids[0,  0] == 1
#         assert token_atcg[0, -1] != 2
#         assert token_atcg[0,  0] != 1

#         # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑  original preprocessing (unchanged)  ‑‑‑‑‑‑‑‑‑‑‑‑
#         initial_len_input_ids = input_ids.shape[1]
#         if initial_len_input_ids < self.max_seq_len:
#             input_ids = np.array(
#                 list(input_ids[0, :]) + [3] * (self.max_seq_len - input_ids.shape[1])
#             ).astype(int).reshape((1, -1))
#         else:
#             input_ids = np.array(
#                 list(input_ids[0, :])[: self.max_seq_len - 1] + [2]
#             ).astype(int).reshape((1, -1))

#         token_type_ids = np.zeros((1, self.max_seq_len))
#         attention_mask = (input_ids != 3).astype(int)

#         atcg_seq = ''
#         token_repeater_numbers = []
#         meaningful_tokens_only = input_ids[0][input_ids[0] > 5]
#         for t in meaningful_tokens_only:
#             atcg_seq_token = self.tokenizer.convert_ids_to_tokens(int(t))
#             token_repeater_numbers.append(len(atcg_seq_token))
#             atcg_seq += atcg_seq_token

#         assert len(atcg_seq) == sum(token_repeater_numbers)

#         token_repeater = []
#         for n, i in enumerate(token_repeater_numbers):
#             for _ in range(i):
#                 token_repeater.append(n)

#         assert len(token_repeater) <= labels.shape[1]

#         labels     = labels[:, : len(token_repeater), :]
#         token_atcg = token_atcg[:, : len(token_repeater)]

#         if len(token_repeater) < self.max_atcg_seq_len:
#             labels = np.concatenate(
#                 (labels, np.full((1, self.max_atcg_seq_len - len(token_repeater), 24), -100)),
#                 axis=1,
#             )
#             token_atcg = np.concatenate(
#                 (token_atcg, np.full((1, self.max_atcg_seq_len - len(token_repeater)), -100)),
#                 axis=1,
#             )
#             token_repeater.extend([-100] * (self.max_atcg_seq_len - len(token_repeater)))
#         else:
#             token_repeater = token_repeater[: self.max_atcg_seq_len]
#             labels     = labels[:, : self.max_atcg_seq_len, :]
#             token_atcg = token_atcg[:, : self.max_atcg_seq_len]

#         token_repeater = np.array(token_repeater).reshape((1, -1)).astype(int)
#         letter_level_mask = token_repeater != -100

#         return {
#             "input_ids": input_ids.squeeze(),
#             "token_type_ids": token_type_ids.squeeze(),
#             "attention_mask": attention_mask.squeeze(),
#             "labels": torch.randint(0, 5, (input_ids.shape[1], 5)),            # change it in future
#             "labels_ohe": torch.randint(0, 5, (input_ids.shape[1], 5)),        # change it in future
#             "labels_mask": (input_ids.squeeze() > 5).astype(int),
#             "letter_level_tokens": token_atcg.squeeze(),
#             "letter_level_labels": labels.squeeze(),
#             "letter_level_labels_mask": letter_level_mask.squeeze(),
#             "embedding_repeater": token_repeater.squeeze(),
#             "letter_level_attention_mask": letter_level_mask.astype(int).squeeze(),
#             "letter_level_token_types_ids": np.zeros(self.max_atcg_seq_len).astype(int),
#         }

#     def close(self):  # everything is already in RAM
#         pass



















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

    def shuffle_starts_func(self, input_ids, labels, token_atcg):
        chooser = np.random.rand()
        # print(chooser)
        if chooser > -1:
            ir_pos_idx = np.arange(labels.shape[1])
            counter = 0
            while True:
                counter += 1
                selected_start = np.random.choice(ir_pos_idx)
                if labels.shape[1] - selected_start > 512: # some threshold
                    break
                if counter > 100:
                    return input_ids, labels, token_atcg

            # print(selected_start, labels.shape)
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

        # sample_name = "transcript_" + str(0)

        input_ids = np.array(
            self.data[sample_name]["token_atcg"]
        )[None, :]
        input_ids = self.tokenizer([self.tokenizer.decode(input_ids[0, :])], return_tensors='np')['input_ids']
        # token_type_ids = np.array(
        #     self.data[sample_name]["token_type_ids"]
        # )
        # attention_mask = np.array(
        #     self.data[sample_name]["attention_mask"]
        # )
        labels = np.array(
            self.data[sample_name]["labels"]
        ).astype(int)[:]#[None, 1:-1, :]#[:]
        new_labels = np.zeros((len(labels), 9))
        new_labels[np.arange(len(labels)), labels] = 1
        labels = new_labels[None, :, :]
        token_atcg = np.array(
            self.data[sample_name]["token_atcg"]
        )[None, :]#[None, :]#[None, 1:-1]
        # print(token_atcg)
        # print(input_ids.shape, labels.shape, token_atcg.shape)
        
        assert token_atcg.shape[1] == labels.shape[1]
        assert input_ids[0, -1] == 2
        assert input_ids[0, 0] == 1
        assert token_atcg[0, -1] != 2
        assert token_atcg[0, 0] != 1

        input_ids, labels, token_atcg = self.shuffle_starts_func(input_ids, labels, token_atcg)

        # input_ids = np.concatenate(([[1]], token_atcg, [[2]]), axis=1)

        
        assert token_atcg.shape[1] == labels.shape[1]
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

        # assert np.all(np.array(token_repeater) == np.arange(len(token_repeater)))
        
        if len(token_repeater) < self.max_atcg_seq_len:
            labels = np.concatenate((labels, np.full((1, self.max_atcg_seq_len - len(token_repeater), 9), -100)), axis=1)
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

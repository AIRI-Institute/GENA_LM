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
        tokenizer,              # original tiny 4-letter tokenizer (for decode only)
        tokenizer_caduseus,     # BPE tokenizer that re-encodes the slice
        tmp_valid_len=None
    ):
        # ────────────────────────────────────────────────────────────────
        # ❶  Load ALL samples into RAM
        # ────────────────────────────────────────────────────────────────
        self.samples_data = {}
        with h5py.File(targets_file, "r") as h5f:
            for k in h5f.keys():                                   # "transcript_42", "sample_007", …
                self.samples_data[k] = {
                    ds: h5f[k][ds][()] for ds in h5f[k].keys()
                }

        # decide dataset type (“sample_” vs “transcript_”) only ONCE
        first_key = next(iter(self.samples_data))
        self.is_sample_dataset = first_key.startswith("sample_")    # new data if True

        self.max_atcg_seq_len   = max_atcg_seq_len
        self.tokenizer          = tokenizer
        self.tokenizer_caduseus = tokenizer_caduseus
        self.tmp_valid_len      = tmp_valid_len

        self.keys    = list(self.samples_data.keys())
        self.samples = len(self.keys) # if len(self.keys) != 3 else 1000


    # ------------------------------------------------------------------
    def __len__(self):
        return self.tmp_valid_len if self.tmp_valid_len is not None else self.samples

    def shuffle_starts_func(self, labels, token_atcg):
        L = labels.shape[0]
        if L <= 512:
            return labels, token_atcg

        while True:
            selected_start = np.random.randint(0, L)  # random start in [0, L)
            if L - selected_start > 512:
                break

        # print(selected_start, len(labels))
        token_atcg = token_atcg[selected_start:]
        labels     = labels[selected_start:]

        # print(selected_start)
        return labels, token_atcg


    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        if self.tmp_valid_len is None:
            assert idx < self.samples

        key = self.keys[idx] if self.tmp_valid_len is None \
              else self.keys[np.random.randint(0, self.samples)]
        sample = self.samples_data[key]
        is_sample = self.is_sample_dataset                             # == key.startswith("sample_")

        # raw arrays (no copies)
        labels_raw      = sample["labels"]                # 1-D int array, shape (L,)
        token_atcg_raw  = sample["token_atcg"]            # 1-D int array, shape (L,)

        # ───────────────────────────────────────────────────────────
        # ❷  If this is a *sample_* dataset, just start from a random
        #     position (no overlap/avoidance mask in this data)
        # ───────────────────────────────────────────────────────────
        if is_sample:
            token_atcg = token_atcg_raw
            labels     = labels_raw
            assert len(labels) > 10
            labels, token_atcg = self.shuffle_starts_func(labels, token_atcg)
        else:
            assert False
            # old “transcript_” data → keep full sequence, then random shift
            token_atcg = token_atcg_raw
            labels     = labels_raw
            assert len(labels) > 10
            labels, token_atcg = self.shuffle_starts_func(labels, token_atcg)



        # ------------------------------------------------------------------
        # original assertions (still valid)
        # ------------------------------------------------------------------
        assert len(token_atcg) == len(labels)
        assert token_atcg[-1] != 2
        assert token_atcg[ 0] != 1

        # ------------------------------------------------------------------
        # ❹  Token-level → string → BPE re-encode  (decode only the slice!)
        # ------------------------------------------------------------------
        atcg_seq = ''.join(list(self.tokenizer.decode(token_atcg[: self.max_atcg_seq_len - 1])))
        labels   = labels[: self.max_atcg_seq_len - 1]
        # atcg_seq = atcg_seq[: self.max_atcg_seq_len - 1]
        token_atcg_bpe = self.tokenizer_caduseus(atcg_seq, return_tensors='np')['input_ids'].squeeze()

        # ------------------------------------------------------------------
        # padding exactly as in the original dataset code
        # ------------------------------------------------------------------
        if labels.shape[0] < self.max_atcg_seq_len:
            labels = np.pad(
                labels,
                pad_width=((self.max_atcg_seq_len - labels.shape[0] - 1, 1)),
                mode='constant',
                constant_values=-100
            )

        if len(token_atcg_bpe) < self.max_atcg_seq_len:
            token_atcg_bpe = np.pad(
                token_atcg_bpe,
                pad_width=((self.max_atcg_seq_len - token_atcg_bpe.shape[0], 0)),
                mode='constant',
                constant_values=4
            )

        assert len(token_atcg_bpe) == len(labels) == self.max_atcg_seq_len

        letter_level_mask = (token_atcg_bpe != 4) & (token_atcg_bpe != 1)

        assert len(token_atcg_bpe[letter_level_mask]) == len(atcg_seq)
        assert len(token_atcg_bpe[letter_level_mask]) == len(labels[letter_level_mask])
        assert -100 not in labels[letter_level_mask]
        assert 1 in token_atcg_bpe
        assert 1 not in token_atcg_bpe[letter_level_mask]

        # ───────────────────────────────────────────────────────────
        # ❺  Labels → one-hot:
        #     - for *sample_* datasets: keep current 3-class setup
        #     - for *transcript_* datasets: remap 24 → 3 (21→1, 22→2, others→0),
        #       assert 21/22 runs have length 1, then widen ±10 and one-hot.
        # ───────────────────────────────────────────────────────────
        class_labels = labels.squeeze().astype(int)
        if is_sample:
            one_hot_labels = np.full((class_labels.shape[0], 9), -100, dtype=np.float32)
            valid_mask = class_labels != -100
            one_hot_labels[valid_mask] = np.eye(9)[class_labels[valid_mask]]
        else:
            assert False
            # remap 24→3
            mask21 = (class_labels == 21)
            mask22 = (class_labels == 22)

            # assert all 21/22 intervals are singletons
            # assert not np.any(mask21[:-1] & mask21[1:]), "Found length>1 interval for label 21"
            # assert not np.any(mask22[:-1] & mask22[1:]), "Found length>1 interval for label 22"

            reduced = np.full(class_labels.shape, 2, dtype=np.int64)
            reduced[mask21] = 0
            reduced[mask22] = 1

            # widen ±10
            N = reduced.shape[0]
            idx21 = np.flatnonzero(mask21)
            idx22 = np.flatnonzero(mask22)

            for i in idx21:
                s = max(i - 10, 0)
                e = min(i + 10, N - 1)
                reduced[s:e + 1] = 0

            for i in idx22:
                s = max(i - 10, 0)
                e = min(i + 10, N - 1)
                reduced[s:e + 1] = 1  

            # one-hot with -100 on invalid (padding) positions
            valid_mask = class_labels != -100
            one_hot_labels = np.full((class_labels.shape[0], 3), -100, dtype=np.float32)
            one_hot_labels[valid_mask] = np.eye(3, dtype=np.float32)[reduced[valid_mask]]

        return {
            "input_ids":               token_atcg_bpe.squeeze(),          # BPE IDs
            "letter_level_labels":     one_hot_labels.astype(float),    # int → float as before
            "letter_level_labels_mask": letter_level_mask.squeeze()
        }

    # ------------------------------------------------------------------
    # Nothing to close; everything is pre-loaded in RAM
    # ------------------------------------------------------------------
    def close(self):
        pass

















# #!/usr/bin/env python
# import h5py
# import numpy as np
# from torch.utils.data import Dataset
# import torch


# class AnnotationDataset(Dataset):
#     def __init__(
#         self,
#         targets_file,
#         max_atcg_seq_len,
#         tokenizer,
#         tokenizer_caduseus,
#         tmp_valid_len = None,
#         shuffle_starts = False
#     ):
#         self.data = h5py.File(targets_file, "r")
#         self.max_atcg_seq_len = max_atcg_seq_len
#         self.tokenizer = tokenizer
#         self.tokenizer_caduseus = tokenizer_caduseus
#         self.tmp_valid_len = tmp_valid_len
#         self.shuffle_starts = shuffle_starts

#         self.samples = len(list(self.data.keys()))

#     def shuffle_starts_func(self, labels, token_atcg):
#         # print('YES')
#         chooser = np.random.rand()
#         # print(chooser)
#         if chooser > 0.5: ###################################################################################################### start upsampling
#             ir_pos_idx = np.arange(labels.shape[0])
#             counter = 0
#             while True:
#                 counter += 1
#                 selected_start = np.random.choice(ir_pos_idx)
#                 if len(labels) - selected_start > 512: # some threshold
#                     break
#                 if counter > 100:
#                     return labels, token_atcg

#             # print(selected_start, len(labels))
#             token_atcg = token_atcg[selected_start:]
#             labels = labels[selected_start:, :]
#         else:
#             assert len(np.argmax(labels, axis=1)) == len(labels)
#             assert 23 in np.argmax(labels, axis=1)
#             ir_pos_idx = np.arange(labels.shape[0])[np.argmax(labels, axis=1)==23]
#             counter = 0
#             while True:
#                 counter += 1
#                 selected_start = np.random.choice(ir_pos_idx)
#                 if (len(labels) - selected_start > 512) and (selected_start < 1500): # some threshold
#                     break
#                 if counter > 100:
#                     return labels, token_atcg

#             # print(selected_start, len(labels))
#             token_atcg = token_atcg[selected_start:]
#             labels = labels[selected_start:, :]
#         return labels, token_atcg

#     def __len__(self):
#         return self.tmp_valid_len if self.tmp_valid_len is not None else self.samples

#     def __getitem__(self, idx):
#         assert idx < self.samples
        
#         if self.tmp_valid_len is None:
#             sample_name = "transcript_" + str(idx)
#         else:
#             sample_name = "transcript_" + str(int(np.random.randint(0, self.samples)))

#         mapper = np.eye(24)
#         labels = np.array(
#             self.data[sample_name]["labels"]
#         )[:]
#         labels = mapper[labels]
 
#         token_atcg = np.array(
#             self.data[sample_name]["token_atcg"]
#         )[:]

#         assert len(token_atcg) == len(labels)
#         assert token_atcg[-1] != 2
#         assert token_atcg[0] != 1

#         labels, token_atcg = self.shuffle_starts_func(labels, token_atcg)

        
#         assert len(token_atcg) == len(labels)
#         assert token_atcg[-1] != 2
#         assert token_atcg[0] != 1

#         atcg_seq = ''.join(list(self.tokenizer.decode(token_atcg)))
#         # print(atcg_seq)
#         labels = labels[:self.max_atcg_seq_len-1, :]
#         atcg_seq = atcg_seq[:self.max_atcg_seq_len-1]
#         token_atcg = self.tokenizer_caduseus(atcg_seq, return_tensors='np')['input_ids'].squeeze()
#         # print(token_atcg)
#         # assert False

#         # print(token_atcg)
#         assert 1 in token_atcg

#         if labels.shape[1] < self.max_atcg_seq_len:
#             labels = np.pad(labels, pad_width=((self.max_atcg_seq_len - labels.shape[0] - 1, 1), (0, 0)), mode='constant', constant_values=-100)

#         if len(token_atcg) < self.max_atcg_seq_len:
#             token_atcg = np.pad(token_atcg, pad_width=((self.max_atcg_seq_len - token_atcg.shape[0], 0)), mode='constant', constant_values=4)

#         assert len(token_atcg) == len(labels)
#         assert len(token_atcg) == self.max_atcg_seq_len

#         letter_level_mask = (token_atcg != 4) & (token_atcg != 1)

#         assert len(token_atcg[letter_level_mask]) == len(atcg_seq)
#         assert len(token_atcg[letter_level_mask]) == len(labels[letter_level_mask, :])
#         assert -100 not in labels[letter_level_mask, :]
#         assert 1 in token_atcg
#         assert 1 not in token_atcg[letter_level_mask]
        
#         return {
#             "input_ids": token_atcg.squeeze(),
#             "letter_level_labels": labels.squeeze().astype(float),
#             "letter_level_labels_mask": letter_level_mask.squeeze()
#         }

#     def close(self):
#         self.data.close()






# #!/usr/bin/env python
# import h5py
# import numpy as np
# from torch.utils.data import Dataset
# import torch

# class AnnotationDataset(Dataset):
#     def __init__(
#         self,
#         targets_file,
#         max_atcg_seq_len,
#         tokenizer,              # original tiny 4‑letter tokenizer (for decode only)
#         tokenizer_caduseus,     # BPE tokenizer that re‑encodes the slice
#         tmp_valid_len=None
#     ):
#         # ────────────────────────────────────────────────────────────────
#         # ❶  Load ALL samples into RAM
#         # ────────────────────────────────────────────────────────────────
#         self.samples_data = {}
#         with h5py.File(targets_file, "r") as h5f:
#             for k in h5f.keys():                                   # "transcript_42", "sample_007", …
#                 self.samples_data[k] = {
#                     ds: h5f[k][ds][()] for ds in h5f[k].keys()
#                 }

#         # decide dataset type (“sample_” vs “transcript_”) only ONCE
#         first_key = next(iter(self.samples_data))
#         self.is_sample_dataset = first_key.startswith("sample_")    # new data if True

#         self.max_atcg_seq_len   = max_atcg_seq_len
#         self.tokenizer          = tokenizer
#         self.tokenizer_caduseus = tokenizer_caduseus
#         self.tmp_valid_len      = tmp_valid_len

#         self.keys    = list(self.samples_data.keys())
#         self.samples = len(self.keys)


#     # ------------------------------------------------------------------
#     def __len__(self):
#         return self.tmp_valid_len if self.tmp_valid_len is not None else self.samples

#     def shuffle_starts_func(self, labels, token_atcg):
#         ir_pos_idx = np.arange(labels.shape[0])
#         counter = 0
#         while True:
#             counter += 1
#             selected_start = np.random.choice(ir_pos_idx)
#             if len(labels) - selected_start > 512:  # some threshold
#                 break
#             if counter > 100:
#                 return labels, token_atcg

#         # print(selected_start, len(labels))
#         token_atcg = token_atcg[selected_start:]
#         labels     = labels[selected_start:]

#         # print(selected_start)
        
#         return labels, token_atcg


#     # ------------------------------------------------------------------
#     def __getitem__(self, idx):
#         assert idx < self.samples
#         key = self.keys[idx] if self.tmp_valid_len is None \
#               else self.keys[np.random.randint(0, self.samples)]
#         sample = self.samples_data[key]
#         is_sample = self.is_sample_dataset                             # == key.startswith("sample_")

#         # raw arrays (no copies)
#         labels_raw      = sample["labels"]                # 1‑D int array, shape (L,)
#         token_atcg_raw  = sample["token_atcg"]             # 1‑D int array, shape (L,)
#         overlap_mask    = sample.get("overlaps", None)     # only in new “sample_” data

#         # ───────────────────────────────────────────────────────────
#         # ❷  If this is a *sample_* dataset, pick one valid segment
#         #     (single contiguous 1‑run ≥512, no stitching)
#         # ───────────────────────────────────────────────────────────
#         if is_sample:
#             if overlap_mask is not None:
#                 overlap_mask = overlap_mask.astype(bool)
#             total_len = token_atcg_raw.shape[0]

#             found = False
#             tries = 0
#             while not found:
#                 tries += 1
#                 # propose a window of size max_atcg_seq_len
#                 start = np.random.randint(0, total_len - self.max_atcg_seq_len + 1)
#                 end   = start + self.max_atcg_seq_len
#                 seg_mask = overlap_mask[start:end]

#                 # disallowed (False) runs
#                 zero_idx = np.where(~seg_mask)[0]
#                 if zero_idx.size == 0:
#                     zero_runs = 0
#                 else:
#                     zsplit = np.split(zero_idx, np.where(np.diff(zero_idx) != 1)[0] + 1)
#                     zero_runs = len([r for r in zsplit if len(r) > 0])

#                 if zero_runs > 1:                       # rule‑2
#                     continue

#                 # allowed (True) runs
#                 ones_idx = np.where(seg_mask)[0]
#                 if ones_idx.size == 0:
#                     continue
#                 osplit = np.split(ones_idx, np.where(np.diff(ones_idx) != 1)[0] + 1)
#                 good_runs = [r for r in osplit if len(r) >= 512]   # rule‑4

#                 if len(good_runs) != 1:                # rule‑3 (no stitching)
#                     continue

#                 run = good_runs[0]
#                 run_s, run_e = run[0], run[-1] + 1     # inclusive → exclusive

#                 token_atcg = token_atcg_raw[run_s + start : run_e + start]
#                 labels     = labels_raw    [run_s + start : run_e + start]
#                 found = True

#                 if tries > 1000:                       # emergency fallback
#                     raise "How that happened?"
#         else:
#             # old “transcript_” data → keep full sequence
#             token_atcg = token_atcg_raw
#             labels     = labels_raw
#             labels, token_atcg = self.shuffle_starts_func(labels, token_atcg)



#         # ------------------------------------------------------------------
#         # original assertions (still valid)
#         # ------------------------------------------------------------------
#         assert len(token_atcg) == len(labels)
#         assert token_atcg[-1] != 2
#         assert token_atcg[ 0] != 1

#         # ------------------------------------------------------------------
#         # ❹  Token‑level → string → BPE re‑encode  (decode only the slice!)
#         # ------------------------------------------------------------------
#         atcg_seq = ''.join(list(self.tokenizer.decode(token_atcg[: self.max_atcg_seq_len - 1])))
#         labels   = labels[: self.max_atcg_seq_len - 1]
#         # atcg_seq = atcg_seq[: self.max_atcg_seq_len - 1]
#         token_atcg_bpe = self.tokenizer_caduseus(atcg_seq, return_tensors='np')['input_ids'].squeeze()

#         # ------------------------------------------------------------------
#         # padding exactly as in the original dataset code
#         # ------------------------------------------------------------------
#         if labels.shape[0] < self.max_atcg_seq_len:
#             labels = np.pad(
#                 labels,
#                 pad_width=((self.max_atcg_seq_len - labels.shape[0] - 1, 1)),
#                 mode='constant',
#                 constant_values=-100
#             )

#         if len(token_atcg_bpe) < self.max_atcg_seq_len:
#             token_atcg_bpe = np.pad(
#                 token_atcg_bpe,
#                 pad_width=((self.max_atcg_seq_len - token_atcg_bpe.shape[0], 0)),
#                 mode='constant',
#                 constant_values=4
#             )

#         assert len(token_atcg_bpe) == len(labels) == self.max_atcg_seq_len

#         letter_level_mask = (token_atcg_bpe != 4) & (token_atcg_bpe != 1)

#         assert len(token_atcg_bpe[letter_level_mask]) == len(atcg_seq)
#         assert len(token_atcg_bpe[letter_level_mask]) == len(labels[letter_level_mask])
#         assert -100 not in labels[letter_level_mask]
#         assert 1 in token_atcg_bpe
#         assert 1 not in token_atcg_bpe[letter_level_mask]

#         # FOR GENERAL 24 CLASSES

#         class_labels = labels.squeeze().astype(int)
#         one_hot_labels = np.full((class_labels.shape[0], 24), -100, dtype=np.float32)
#         valid_mask = class_labels != -100
#         one_hot_labels[valid_mask] = np.eye(24)[class_labels[valid_mask]]

#         return {
#             "input_ids":               token_atcg_bpe.squeeze(),          # BPE IDs
#             "letter_level_labels":     one_hot_labels.astype(float),    # int → float as before
#             "letter_level_labels_mask": letter_level_mask.squeeze()
#         }

#     # ------------------------------------------------------------------
#     # Nothing to close; everything is pre‑loaded in RAM
#     # ------------------------------------------------------------------
#     def close(self):
#         pass
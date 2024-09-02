from typing import Optional

import h5py
import numpy as np
from Bio.Seq import Seq
from torch.utils.data import Dataset


class EnformerDataset(Dataset):
    """
    seq = left_context; BIN_SIZE x BINS_COUNT; right_context
    target: BINS_COUNT x TG_COUNT

    EnformerDataset splits seq and target on chunks with n_bins = `bins_per_sample`:
    ->
    chunk_seq := CLS bin_1 SEP bin_2 SEP bin_3 ... SEP
    target: bins_per_sample x TG_COUNT
    """
    TG_COUNT = 5313
    BINS_COUNT = 896
    BIN_SIZE = 128
    PAD = (196608 - BIN_SIZE * BINS_COUNT) // 2
    AUG_SHIFT = [-3, -2, -1, 0, 1, 2, 3]

    def __init__(self,
                 tokenizer,
                 path: str,
                 max_seq_len: int = 512,
                 bins_per_sample: int = BINS_COUNT,
                 n_context_bins: int = 0,
                 remove_context: bool = True,
                 context_mode: Optional[str] = None,
                 augment: bool = False,
                 aug_shift_max: int = 3,
                 aug_shift_min: int = -3,
                 aug_rc: str = 'rnd',
                 n_samples=None):
        """
        Args:
            aug_rc (str, optional): 'keep' to keep sequence as is, 'rc' to replace sequence with reverse complementary,
                'rnd' to sample randomly to keep or to reverse complement. Defaults to 'rnd'.
        """
        self.h5_file = h5py.File(path, "r")
        self.h5_keys = np.asarray(list(self.h5_file.keys()))
        if n_samples and n_samples > 0:
            self.h5_keys = self.h5_keys[:n_samples]

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.bins_per_sample = bins_per_sample
        self.remove_context = remove_context
        self.n_context_bins = n_context_bins
        self.context_mode = context_mode

        self.n_center_bins = bins_per_sample - n_context_bins
        self.n_chunks = int(np.ceil(self.BINS_COUNT / self.n_center_bins))
        self.n_records = len(self.h5_keys)

        # augmentation attributes
        self.augment = augment
        self.AUG_SHIFT = list(range(aug_shift_min, aug_shift_max + 1))
        self.aug_rc = aug_rc

    def __len__(self):
        # each record in dataset is split on n_chunks, in total we have n_records x n_chunks samples in a dataset
        return self.n_records * self.n_chunks

    @staticmethod
    def coin_flip():
        return np.random.random() > 0.5

    @staticmethod
    def reverse_complement_bins(bins):
        rc_bins = []
        for bin in bins:
            rc_bins += [str(Seq(bin).reverse_complement())]
        return rc_bins[::-1]

    def __getitem__(self, idx):
        record_idx = idx // self.n_chunks
        chunk_idx = idx % self.n_chunks
        k = self.h5_keys[record_idx]

        # it takes ~200ms to read single seq & target from h5file (in random order)
        seq = self.h5_file[k]['seq'][()].decode('utf-8')
        # read only bins for current sample
        target = self.h5_file[k]['target'][chunk_idx * self.n_center_bins: (chunk_idx + 1) * self.n_center_bins, :]

        left_pad = EnformerDataset.PAD
        right_pad = EnformerDataset.PAD
        if self.augment:
            # add random shift
            shift = np.random.choice(self.AUG_SHIFT)
            left_pad = EnformerDataset.PAD + shift
            right_pad = EnformerDataset.PAD - shift

        center = seq[left_pad: -right_pad]

        if self.remove_context or self.n_context_bins == 0:
            left, right = '', ''
        else:
            left, right = seq[:left_pad], seq[-right_pad:]

        assert len(center) // self.BIN_SIZE == self.BINS_COUNT

        # collect center bins
        bins_seq = []
        for i in range(self.n_center_bins):
            bin_st = (chunk_idx * self.n_center_bins + i) * self.BIN_SIZE
            bin_end = (chunk_idx * self.n_center_bins + i + 1) * self.BIN_SIZE
            if bin_st < len(center):
                bins_seq += [center[bin_st:bin_end]]

        # collect context bins (depends on reverse?)
        center_bins_st = (chunk_idx * self.n_center_bins) * self.BIN_SIZE
        center_bins_end = min(bin_end, len(center))
        # concat left/right pad sequence with left/right remainder of center sequence
        left = left + center[:center_bins_st]
        right = center[center_bins_end:] + right
        # get left/right context sequences of needed length
        left = left[-(self.n_context_bins * self.BIN_SIZE):]
        right = right[:(self.n_context_bins * self.BIN_SIZE)]
        left_bins_seq = [left[::-1][i: i+self.BIN_SIZE:][::-1] for i in range(0, len(left), self.BIN_SIZE)][::-1]
        right_bins_seq = [right[i: i+self.BIN_SIZE:] for i in range(0, len(right), self.BIN_SIZE)]

        # reverse complement
        reverse_complement = False
        if self.augment and ((self.coin_flip() and self.aug_rc == 'rnd') or self.aug_rc == 'rc'):
            bins_seq = self.reverse_complement_bins(bins_seq)
            target = np.flip(target, axis=0)
            reverse_complement = True

        encoded_bins = self.tokenizer.batch_encode_plus(bins_seq, add_special_tokens=False, return_attention_mask=False,
                                                        return_token_type_ids=False)['input_ids']

        encoded_left_bins, encoded_right_bins = None, None
        if self.context_mode is None:
            ...
        elif self.context_mode == 'left':
            if reverse_complement:
                left_bins_seq = self.reverse_complement_bins(right_bins_seq)
            if len(left_bins_seq) > 0:
                encoded_left_bins = self.tokenizer.batch_encode_plus(left_bins_seq, add_special_tokens=False,
                                                                     return_attention_mask=False,
                                                                     return_token_type_ids=False)['input_ids']
        else:
            raise NotImplementedError

        # CLS left_bin_1 SEP left_bin_2 SEP bin_1 SEP bin_2 SEP bin_3 SEP ... bin_n SEP right SEP
        sample_token_ids = [self.tokenizer.cls_token_id]
        bins_mask = [0]
        n_center_bins = 0
        for bin_token_ids in encoded_bins:
            if len(sample_token_ids) + len(bin_token_ids) + 1 < self.max_seq_len:
                sample_token_ids += bin_token_ids + [self.tokenizer.sep_token_id]
                bins_mask += [0] * len(bin_token_ids) + [1]
                n_center_bins += 1

        bins_mask = (np.array(sample_token_ids) == self.tokenizer.sep_token_id).astype(bool)
        # insert left context
        n_left_bins = 0
        if encoded_left_bins is not None:
            for bin_token_ids in encoded_left_bins[::-1]:
                if len(sample_token_ids) + len(bin_token_ids) + 1 < self.max_seq_len:
                    sample_token_ids = sample_token_ids[:1] + bin_token_ids + [self.tokenizer.sep_token_id] + sample_token_ids[1:]
                    n_left_bins += 1

        # mask for CLS - 0, left bins - 0, center bins - 1
        sample_token_ids = np.array(sample_token_ids)
        bins_mask = np.concatenate([np.zeros(len(sample_token_ids) - len(bins_mask), dtype=bool), bins_mask], axis=0)
        token_type_ids = np.zeros(len(sample_token_ids), dtype=np.int64)
        attention_mask = np.ones(len(sample_token_ids), dtype=np.int64)

        # labels = target[chunk_idx * self.bins_per_sample:(chunk_idx + 1) * self.bins_per_sample]
        # take labels for bins that fully fit into the sample_token_ids
        labels = target[:n_center_bins, :]
        assert labels.shape[0] == bins_mask.sum()

        return {'input_ids': sample_token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'bins_mask': bins_mask,
                'labels': labels}

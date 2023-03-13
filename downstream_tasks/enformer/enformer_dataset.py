import h5py
import numpy as np
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

    def __init__(self, tokenizer, path: str, max_seq_len: int = 512, bins_per_sample: int = BINS_COUNT,
                 remove_context=True, n_samples=None):
        self.h5_file = h5py.File(path, "r")
        self.h5_keys = np.asarray(list(self.h5_file.keys()))
        if n_samples and n_samples > 0:
            self.h5_keys = self.h5_keys[:n_samples]

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.bins_per_sample = bins_per_sample
        self.remove_context = remove_context

        self.n_chunks = int(np.ceil(self.BINS_COUNT / self.bins_per_sample))
        self.n_records = len(self.h5_keys)

    def __len__(self):
        # each record in dataset is split on n_chunks, in total we have n_records x n_chunks samples in a dataset
        return self.n_records * self.n_chunks

    def __getitem__(self, idx):
        record_idx = idx // self.n_chunks
        chunk_idx = idx % self.n_chunks
        k = self.h5_keys[record_idx]

        # it takes ~200ms to read single seq & target from h5file (in random order)
        seq = self.h5_file[k]['seq'][()].decode('utf-8')
        # read only bins for current sample
        target = self.h5_file[k]['target'][chunk_idx * self.bins_per_sample: (chunk_idx + 1) * self.bins_per_sample, :]

        center = seq[EnformerDataset.PAD: -EnformerDataset.PAD]
        if self.remove_context:
            left, right = '', ''
        else:
            raise NotImplementedError

        assert len(center) // self.BIN_SIZE == self.BINS_COUNT

        # collect bins
        bins_seq = []
        for i in range(self.bins_per_sample):
            bin_st = (chunk_idx * self.bins_per_sample + i) * self.BIN_SIZE
            bin_end = (chunk_idx * self.bins_per_sample + i + 1) * self.BIN_SIZE
            if bin_st < len(center):
                bins_seq += [center[bin_st:bin_end]]

        encoded_bins = self.tokenizer.batch_encode_plus(bins_seq, add_special_tokens=False, return_attention_mask=False,
                                                        return_token_type_ids=False)['input_ids']

        # CLS left SEP bin_1 SEP bin_2 SEP bin_3 SEP ... bin_n SEP right SEP
        sample_token_ids = [self.tokenizer.cls_token_id]
        for bin_token_ids in encoded_bins:
            if len(sample_token_ids) + len(bin_token_ids) + 1 < self.max_seq_len:
                sample_token_ids += bin_token_ids + [self.tokenizer.sep_token_id]

        sample_token_ids = np.array(sample_token_ids)
        token_type_ids = np.array([0] * len(sample_token_ids))
        attention_mask = np.array([1] * len(sample_token_ids))
        bins_mask = (sample_token_ids == self.tokenizer.sep_token_id).astype(bool)

        # labels = target[chunk_idx * self.bins_per_sample:(chunk_idx + 1) * self.bins_per_sample]
        # take labels for bins that fully fit into the sample_token_ids
        labels = target[:bins_mask.sum(), :]

        return {'input_ids': sample_token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'bins_mask': bins_mask,
                'labels': labels}

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EPDnewPromoterDataset(Dataset):
    def __init__(self, datafiles, tokenizer, x_field='x', label_field='label', max_seq_len=512, pad_to_max=True,
                 truncate='right'):
        if isinstance(datafiles, str):
            # convert str path to folder to Path
            datafiles = Path(datafiles)
        if isinstance(datafiles, Path) and datafiles.is_dir():
            # get all files from folder
            datafiles = list(datafiles.iterdir())
        self.data = pd.DataFrame()
        for f in datafiles:
            self.data = pd.concat([self.data, pd.read_csv(f)])
        self.data = self.data.reset_index()
        self.x_field = x_field
        self.label_field = label_field
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_to_max = pad_to_max
        self.truncate = truncate

    @staticmethod
    def get_features(x, tokenizer, max_seq_len=512, pad_to_max=True, truncate='right'):
        tokens = tokenizer.tokenize(x)
        if truncate == 'right':
            tokens = tokens[:max_seq_len-2]
        elif truncate == 'left':
            tokens = tokens[-(max_seq_len-2):]
        elif truncate == 'mid':
            mid = len(tokens) // 2
            left_ctx = (max_seq_len-2) // 2
            right_ctx = (max_seq_len-2) - left_ctx
            tokens = tokens[max(0, mid - left_ctx): min(mid + right_ctx, len(tokens))]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        seq_len = len(tokens)
        token_type_ids = [0] * seq_len
        attention_mask = [1] * seq_len
        if pad_to_max:
            input_ids += [tokenizer.pad_token_id] * max(max_seq_len - seq_len, 0)
            token_type_ids += [0] * max(max_seq_len - seq_len, 0)
            attention_mask += [0] * max(max_seq_len - seq_len, 0)
        return {'input_ids': np.array(input_ids),
                'token_type_ids': np.array(token_type_ids),
                'attention_mask': np.array(attention_mask)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[self.x_field][idx]
        features = EPDnewPromoterDataset.get_features(x, self.tokenizer, self.max_seq_len, self.pad_to_max,
                                                      self.truncate)
        label = {'labels': self.data[self.label_field][idx]}
        return {**features, **label}

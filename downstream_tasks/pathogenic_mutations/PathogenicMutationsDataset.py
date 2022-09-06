import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.gena_lm.utils import concatenate_encodings, symmetric_pad_and_truncate_context


class PathogenicMutationsDataset(Dataset):
    def __init__(
        self,
        data_file,
        tokenizer,
        mid_token="[MASK]",  # or 'Ref', 'Alt' -- which token to put in the middle
        max_seq_len=512,
    ):
        if isinstance(data_file, pd.DataFrame):
            self.data = data_file
        else:
            self.data = pd.read_csv(data_file, sep="\t")
        self.tokenizer = tokenizer

        self.SEP_token_id = self.tokenizer.sep_token_id
        self.MASK_token_id = self.tokenizer.mask_token_id
        self.CLS_token_id = self.tokenizer.cls_token_id
        self.PAD_token_id = self.tokenizer.pad_token_id

        self.CLS_encoding = {
            "input_ids": np.array(self.CLS_token_id).reshape(-1, 1),
            "token_type_ids": np.array([0]).reshape(-1, 1),
            "attention_mask": np.array([1]).reshape(-1, 1),
        }

        self.SEP_encoding = {
            "input_ids": np.array(self.SEP_token_id).reshape(-1, 1),
            "token_type_ids": np.array([0]).reshape(-1, 1),
            "attention_mask": np.array([1]).reshape(-1, 1),
        }

        self.mid_token = mid_token
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        ref = sample["Ref"].upper()
        alt = sample["Alt"].upper()
        label = sample["Label"]

        # tokenize to the form:
        # final form should be: CLS-left-MASK-right-SEP-PAD
        n_service_tokens = 2

        left_tokens = self.tokenizer(
            sample["left"],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_tensors="np",
        )

        right_tokens = self.tokenizer(
            sample["right"],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_tensors="np",
        )

        if self.mid_token == "[MASK]":
            mid_token_id = self.MASK_token_id
        else:
            mid_token_id = self.tokenizer.convert_tokens_to_ids(sample[self.mid_token])

        mid_tokens = {
            "input_ids": np.array([[mid_token_id]]),
            "token_type_ids": np.array([[0]]),
            "attention_mask": np.array([[1]]),
        }

        (
            left_encoding,
            right_encoding,
            mid_encoding,
            padding,
        ) = symmetric_pad_and_truncate_context(
            left_encoding=left_tokens,
            mid_encoding=mid_tokens,
            right_encoding=right_tokens,
            n_service_tokens=n_service_tokens,
            max_seq_len=self.max_seq_len,
            PAD_id=self.PAD_token_id,
        )

        return concatenate_encodings(
            [
                self.CLS_encoding,
                left_encoding,
                mid_encoding,
                right_encoding,
                self.SEP_encoding,
                padding,
            ]
        )

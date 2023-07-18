import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SpliceAIDataset(Dataset):
    def __init__(
        self,
        datafile,
        tokenizer,
        max_seq_len=512,
        targets_offset=5000,
        targets_len=5000,
    ):
        self.data = pd.read_csv(datafile, sep=',', header=None)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.targets_offset = targets_offset
        self.targets_len = targets_len

        self.data['targets'] = list(self.data.iloc[:, 1:].values)
        self.data = self.data[[0, 'targets']]
        self.data.columns = ['seq', 'targets']

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_token_classes(seq_encoding, target, targets_offset):
        tokens_info = pd.DataFrame.from_records(
            seq_encoding["offset_mapping"][0], columns=["st", "en"]
        )
        tokens_info["length"] = tokens_info["en"] - tokens_info["st"]

        # handle special tokens which can be anywhere in the seq with the offeset (0,0)
        nonzero_length_mask = tokens_info["length"].values > 0

        # all other tokens should have ascending token_start coordinate
        assert np.all(
            tokens_info[nonzero_length_mask]["st"].values[:-1]
            <= tokens_info[nonzero_length_mask]["st"].values[1:]
        )

        # fill target class information
        target_field_names = []
        for target_class in [1, 2]:
            target_field_name = "class_" + str(target_class)
            target_field_names.append(target_field_name)
            tokens_info[target_field_name] = 0

            nonzero_target_positions = (
                np.where(target == target_class)[0] + targets_offset
            )  # non-zero target coordinates
            nonzero_target_token_ids = (
                np.searchsorted(
                    tokens_info[nonzero_length_mask]["st"],
                    nonzero_target_positions,
                    side="right",
                )
                - 1
            )  # ids of tokens
            # containing non-zero targets
            # in sequence coordinate system
            nonzero_target_token_ids = (
                tokens_info[nonzero_length_mask].iloc[nonzero_target_token_ids].index
            )
            tokens_info.loc[nonzero_target_token_ids, target_field_name] = target_class
            # tokens_info.loc[nonzero_target_token_ids, target_field_name] = 1
            # fill all service tokens with -100
            tokens_info.loc[~nonzero_length_mask, target_field_name] = -100

            # fill context tokens with -100
            target_first_token = (
                np.searchsorted(
                    tokens_info[nonzero_length_mask]["st"],
                    targets_offset,
                    side="right",
                )
                - 1
            )

            mask_ids = (
                tokens_info.loc[nonzero_length_mask, :].iloc[:target_first_token].index
            )
            tokens_info.loc[mask_ids, target_field_name] = -100

            target_last_token = (
                np.searchsorted(
                    tokens_info[nonzero_length_mask]["st"],
                    targets_offset + len(target) - 1,
                    side="right",
                )
                - 1
            )
            if target_last_token + 1 < len(
                tokens_info.loc[nonzero_length_mask, target_field_name]
            ):
                target_last_token += 1
                mask_ids = tokens_info.loc[nonzero_length_mask, :][target_last_token:].index
                tokens_info.loc[mask_ids, target_field_name] = -100
        return tokens_info[target_field_names].values

    def tokenize_inputs(self, seq, target):
        depad_seq_l = seq.lstrip("N")
        if self.targets_offset - (len(seq) - len(depad_seq_l)) < 0:
            depad_seq_l = seq[self.targets_offset:]
        targets_offset = self.targets_offset - (len(seq) - len(depad_seq_l))
        assert targets_offset >= 0
        assert targets_offset + self.targets_len <= len(depad_seq_l)

        depad_seq_both = depad_seq_l.strip("N")
        if targets_offset + self.targets_len > len(depad_seq_both):
            seq = depad_seq_l[:targets_offset + self.targets_len]
        else:
        	seq = depad_seq_both

        left, mid, right = (
            seq[:targets_offset],
            seq[targets_offset : targets_offset + self.targets_len],
            seq[targets_offset + self.targets_len :],
        )

        mid_encoding = self.tokenizer(
            mid,
            add_special_tokens=False,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        context_encoding = self.tokenizer(
            left + "X" + right,
            add_special_tokens=False,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="np",
        )

        for encoding in [mid_encoding, context_encoding]:
            assert np.all(encoding["attention_mask"][0] == 1)
            assert np.all(encoding["token_type_ids"][0] == 0)

        labels = self.get_token_classes(mid_encoding, target, 0)
        token_type_ids = np.zeros(shape=self.max_seq_len, dtype=np.int64)

        boundary_pos = int(
            np.where(
                context_encoding["offset_mapping"][0] == [len(left), len(left) + 1]
            )[0][0]
        )
        boundary_token = context_encoding["input_ids"][0][boundary_pos].tolist()
        assert (
            self.tokenizer.convert_ids_to_tokens(boundary_token) == "[UNK]"
        ), "Error during context tokens processing"

        n_service_tokens = 4  # CLS-left-SEP-mid-SEP-right-SEP (PAD)

        L_mid = len(mid_encoding["input_ids"][0])
        L_left = boundary_pos
        L_right = len(context_encoding["token_type_ids"][0]) - L_left - 1

        # case I. target's encoding >= max_seq_len; don't add context & trim target if needed
        if L_mid + n_service_tokens >= self.max_seq_len:
            # st = (L_mid // 2) - (self.max_seq_len - n_service_tokens) // 2
            # en = st + (self.max_seq_len - n_service_tokens)
            st = 0
            en = self.max_seq_len - n_service_tokens

            input_ids = np.concatenate(
                [
                    [
                        self.tokenizer.convert_tokens_to_ids("[CLS]"),
                        self.tokenizer.convert_tokens_to_ids("[SEP]"),
                    ],
                    mid_encoding["input_ids"][0][st:en],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")] * 2,
                ]
            )
            labels = np.concatenate(
                [[[-100, -100]] * 2, labels[st:en], [[-100, -100]] * 2]
            )
        # case II. target+context encoding < max_seq_len, we need to pad
        elif L_mid + L_left + L_right + n_service_tokens <= self.max_seq_len:
            n_pads = self.max_seq_len - (L_mid + L_left + L_right + n_service_tokens)
            input_ids = np.concatenate(
                [
                    [self.tokenizer.convert_tokens_to_ids("[CLS]")],
                    context_encoding["input_ids"][0][:boundary_pos],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    mid_encoding["input_ids"][0],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    context_encoding["input_ids"][0][boundary_pos + 1:],
                    [self.tokenizer.convert_tokens_to_ids("[PAD]")] * n_pads,
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                ]
            )
            labels = np.concatenate(
                [
                    [[-100, -100]],
                    [[-100, -100]] * (boundary_pos + 1),
                    labels,
                    [[-100, -100]]
                    * (len(context_encoding["input_ids"][0]) - boundary_pos -1 + 1),
                    [[-100, -100]] * (n_pads + 1)
                ]
            )
        # case III. target+context encoding > max_seq_len, we need to trim
        elif L_mid + L_left + L_right + n_service_tokens > self.max_seq_len:
            # compute trimming. The aims are to
            # a) make the total length == self.max_seq_len
            # b) make the left and right context size as close to each other as possible
            oversize = L_mid + L_left + L_right + n_service_tokens - self.max_seq_len
            if L_left >= L_right:
                trim_left = oversize / 2.0 + min(
                    (L_left - L_right) / 2.0, oversize / 2.0
                )
                trim_right = max(0, (oversize - (L_left - L_right)) / 2.0)
            else:
                trim_right = oversize / 2.0 + min(
                    (L_right - L_left) / 2.0, oversize / 2.0
                )
                trim_left = max(0, (oversize - (L_right - L_left)) / 2.0)
            assert (int(trim_right) == trim_right) == (int(trim_left) == trim_left)
            if int(trim_right) != trim_right:
                trim_left += 0.5
                trim_right -= 0.5
            assert (int(trim_right) - trim_right) == (int(trim_left) - trim_left) == 0
            assert oversize == trim_left+trim_right

            trim_left = int(trim_left)
            trim_right = int(trim_right)

            input_ids = np.concatenate(
                [
                    [self.tokenizer.convert_tokens_to_ids("[CLS]")],
                    context_encoding["input_ids"][0][trim_left:boundary_pos],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    mid_encoding["input_ids"][0],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    context_encoding["input_ids"][0][
                        boundary_pos + 1 : L_left + L_right + 1 - trim_right
                    ],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                ]
            )
            labels = np.concatenate(
                [
                    [[-100, -100]],
                    [[-100, -100]] * (boundary_pos - trim_left + 1),
                    labels,
                    [[-100, -100]],
                    [[-100, -100]] * (L_right - trim_right + 1),
                ]
            )
        else:
            raise ValueError("Unexpected encoding length")

        assert len(input_ids) == len(labels) == self.max_seq_len

        # convert labels to (seq_len, n_labels) shape
        n_labels = 3
        labels_ohe = np.zeros((len(labels), n_labels))
        for label in range(n_labels):
            labels_ohe[(labels == label).max(axis=-1), label] = 1.0

        # set mask to 0.0 for tokens with no labels, these examples should not be used for loss computation
        labels_mask = np.ones(len(labels))
        labels_mask[labels_ohe.sum(axis=-1) == 0.0] = 0.0

        attention_mask = np.array(input_ids!=self.tokenizer.pad_token_id, 
                                  dtype=np.int64
                                )
        return {
            "input_ids": input_ids.astype(np.int64),
            "token_type_ids": token_type_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "labels": labels,
            "labels_ohe": labels_ohe,
            "labels_mask": labels_mask,
        }

    def __getitem__(self, idx):
        # tokenize seq
        seq, target = self.data.iloc[idx].values
        assert self.targets_len==len(target)
        return self.tokenize_inputs(seq, target)

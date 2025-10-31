import torch


class ExpressionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        pad_keys = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "labels_mask",
        ]
        no_pad_keys = ["desc_vectors", "tpm", "dataset_mean", "dataset_deviation"]
        special_keys = ["gene_id", "selected_keys", "dataset_description"]

        pad_token_ids = {
            "input_ids": self.tokenizer.pad_token_id,
            "token_type_ids": 0,
            "attention_mask": 0,
            "labels": 0,
            "labels_mask": 0,
        }

        max_seq_len = max([len(sample["input_ids"]) for sample in batch])
        batch_dict = {key: [] for key in pad_keys + no_pad_keys + special_keys}

        for sample in batch:
            for key in pad_keys:
                seq = sample[key]
                pad_len = max_seq_len - len(seq)
                if pad_len > 0:
                    if key in ["labels", "labels_mask"]:
                        pad = torch.full(
                            (pad_len, seq.size(1)), pad_token_ids[key], dtype=seq.dtype
                        )
                        padded_seq = torch.cat([seq, pad], dim=0)
                    else:
                        pad = torch.full(
                            (pad_len,), pad_token_ids[key], dtype=seq.dtype
                        )
                        padded_seq = torch.cat([seq, pad], dim=0)
                else:
                    padded_seq = seq
                batch_dict[key].append(padded_seq)

            for key in no_pad_keys:
                batch_dict[key].append(sample[key])

            for key in special_keys:
                batch_dict[key].append(sample[key])

        for key in pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key])
        for key in no_pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key])
        # special_keys оставляем списками (строки/списки)

        return batch_dict

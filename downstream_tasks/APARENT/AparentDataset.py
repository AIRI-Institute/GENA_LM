import pandas as pd
from torch.utils.data import Dataset


class AparentDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        max_seq_len=512,
    ):
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        results = self.tokenizer(
            text=sample["seq"],
            text_pair=sample["seq_ext"],
            add_special_tokens=True,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_seq_len,
            return_tensors="np",
        )
        targets = sample["target"]
        return {
            "input_ids": results["input_ids"][0],
            "token_type_ids": results["token_type_ids"][0],
            "attention_mask": results["attention_mask"][0],
            "targets": targets,
        }

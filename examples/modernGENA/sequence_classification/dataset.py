from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class GenomicSequenceClassificationDataset(Dataset):
    """Simple CSV-backed dataset for DNA sequence classification."""

    def __init__(self, data_path: str, tokenizer, max_length: int):
        self.data_path = Path(data_path).expanduser().absolute()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(self.data_path)
        required_columns = {"sequence", "label"}
        missing = required_columns.difference(self.data.columns)
        if missing:
            raise ValueError(
                f"Dataset at {self.data_path} is missing columns: {sorted(missing)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        encoded = self.tokenizer(
            row["sequence"],
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        encoded["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return encoded

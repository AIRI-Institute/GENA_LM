from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import pysam
import torch
from torch.utils.data import Dataset


class CTCFTokenRegressionDataset(Dataset):
    """On-the-fly token-level regression dataset using FASTA + bigWig."""

    def __init__(
        self,
        intervals_csv: str,
        fasta_path: str,
        bigwig_path: str,
        tokenizer,
        max_length: int = 1024,
        gap_token_id: int = 5,
    ):
        self.intervals_csv = Path(intervals_csv).expanduser().absolute()
        self.fasta_path = Path(fasta_path).expanduser().absolute()
        self.bigwig_path = Path(bigwig_path).expanduser().absolute()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.gap_token_id = gap_token_id

        self.data = pd.read_csv(self.intervals_csv)
        required = {"chrom", "start", "end"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.intervals_csv}: {sorted(missing)}")

        self._fasta = None
        self._bw = None

    def _ensure_handles(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.fasta_path))
        if self._bw is None:
            self._bw = pyBigWig.open(str(self.bigwig_path))

    def __len__(self):
        return len(self.data)

    def _fix_gap_offsets(self, input_ids: torch.Tensor, offsets: list[list[int]]):
        for idx, token_id in enumerate(input_ids.tolist()):
            if token_id == self.gap_token_id and idx > 0:
                # Gap token can collapse a run of Ns into one token; align its start
                # with the previous token end to keep token spans monotonic.
                offsets[idx][0] = offsets[idx - 1][1]

    def __getitem__(self, idx: int):
        self._ensure_handles()

        row = self.data.iloc[idx]
        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])

        seq = self._fasta.fetch(chrom, start, end).upper()
        encoding = self.tokenizer(
            seq,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        offsets = [list(x) for x in encoding["offset_mapping"]]
        self._fix_gap_offsets(input_ids, offsets)

        signal = np.array(self._bw.values(chrom, start, end, numpy=True), dtype=np.float32)
        signal = np.nan_to_num(signal, nan=0.0)

        labels = torch.full((len(offsets),), -100.0, dtype=torch.float32)
        loss_mask = torch.zeros((len(offsets),), dtype=torch.float32)

        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if attention_mask[tok_idx].item() == 0:
                continue
            if tok_start > tok_end:
                raise ValueError(f"Invalid token offset span [{tok_start}, {tok_end}) for signal length {len(signal)}.")
            if tok_start == tok_end:
                # Some tokens (e.g., special/pad, or adjusted gap-aligned spans) may
                # have empty offsets, so there is no bp window to average.
                continue
            lo = tok_start
            hi = tok_end
            assert (
                0 <= lo < hi <= len(signal)
            ), f"Invalid token offset span [{lo}, {hi}) for signal length {len(signal)}."
            labels[tok_idx] = float(signal[lo:hi].mean())
            loss_mask[tok_idx] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
        }

    def __del__(self):
        if self._fasta is not None:
            self._fasta.close()
        if self._bw is not None:
            self._bw.close()

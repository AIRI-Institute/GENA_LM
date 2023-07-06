import pandas as pd
from torch.utils.data import Dataset

class DeepSTARRDataset(Dataset):
    def __init__(
        self,
        targets_file,
        fasta_file,
        tokenizer,
        max_seq_len=512,
    ):
        self.data = pd.read_csv(targets_file, sep="\t")[
            ["Dev_log2_enrichment", "Hk_log2_enrichment"]
        ]
        seqs = []
        with open(fasta_file) as fin:
            header = False
            for line in fin:
                l = line.strip()
                if len(l) == 0:  # last line
                    break
                if line.startswith(">"):
                    header = True
                    continue
                else:
                    assert header  # check fasta format is correct
                    seqs.append(l)
        assert len(seqs) == len(
            self.data
        ), "Number of targets does not match number of sequences"
        self.data["seq"] = seqs

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        results = self.tokenizer(
            sample["seq"],
            add_special_tokens=True,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_seq_len,
            return_tensors="np",
        )
        targets = sample[["Dev_log2_enrichment", "Hk_log2_enrichment"]].values.astype("float32")
        return {
            "input_ids": results["input_ids"][0],
            "token_type_ids": results["token_type_ids"][0],
            "attention_mask": results["attention_mask"][0],
            "labels": targets,
        }

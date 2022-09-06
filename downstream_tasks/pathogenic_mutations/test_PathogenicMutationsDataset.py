import random

import tqdm
from transformers import AutoTokenizer

from downstream_tasks.pathogenic_mutations.PathogenicMutationsDataset import (
    PathogenicMutationsDataset,
)


class TestPathogenicMutationsDataset:
    def test_sequence_tokenization(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "data/tokenizers/t2t_1000h_multi_32k/",
        )
        datafile = "downstream_tasks/pathogenic_mutations/test_dataset.tsv"

        for max_seq_length in tqdm.tqdm([10, 100, 256, 512, 1024, 2048, 32000]):
            self.dataset = PathogenicMutationsDataset(
                datafile, tokenizer, max_seq_len=max_seq_length
            )
            for i in range(len(self.dataset)):
                tokens = self.dataset.__getitem__(i)
                seq = self.dataset.data.iloc[i].left + self.dataset.data.iloc[i].right
                decoded = tokenizer.decode(
                    tokens["input_ids"][0], skip_special_tokens=True
                )
                assert seq.find(decoded) != -1


# TestPathogenicMutationsDataset().test_sequence_tokenization()

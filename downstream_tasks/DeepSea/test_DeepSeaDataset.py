import tqdm
from downstream_tasks.DeepSea.DeepSeaDataset import DeepSeaDataset
from transformers import AutoTokenizer


class TestDeepSeaDataset:
    def set_variables(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "data/tokenizers/t2t_1000h_multi_32k/",
        )
        self.datafile = "downstream_tasks/DeepSea/test.dataset.csv.gz"

    def test_different_seqlengths(self):
        self.set_variables()
        for max_seq_length in tqdm.tqdm([10, 100, 256, 512, 1024, 2048, 32000]):
            dataset = DeepSeaDataset(self.datafile, self.tokenizer, max_seq_len=max_seq_length)

            for ind in range(len(dataset)):
                dataset[ind]

    def print_sample(self, seq_length):
        self.set_variables()
        dataset = DeepSeaDataset(self.datafile, self.tokenizer, max_seq_len=seq_length)
        sample = dataset.__getitem__(0)
        long_sample = self.tokenizer(
            dataset.data.seq.values[0], add_special_tokens=True, padding=True, truncation=False, return_tensors="np"
        )

        print("short:\n", " ".join(map(str, sample["input_ids"])))
        print("long:\n", " ".join(map(str, long_sample["input_ids"][0])))
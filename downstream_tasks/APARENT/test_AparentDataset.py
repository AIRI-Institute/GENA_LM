from transformers import AutoTokenizer

from AparentDataset import AparentDataset


class TestAparentDataset:
    def test_getitem(self):
        tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base")
        datafile = "downstream_tasks/APARENT/AparentDataset_test_data.csv"

        seq_len = 250

        dataset = AparentDataset(
            path=datafile,
            tokenizer=tokenizer,
            max_seq_len=seq_len,
        )
        res = dataset.__getitem__(2)
        assert len(res["input_ids"]) == seq_len

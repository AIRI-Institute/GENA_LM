from transformers import AutoTokenizer

from DeepSTARRDataset import DeepSTARRDataset


class TestDeepSTARRDataset:
    def test_getitem(self):
        tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base")
        datafile = "downstream_tasks/DeepSTARR/test_Sequences_activity.txt"
        fastafile = "downstream_tasks/DeepSTARR/test_Sequences.fa"

        seq_len = 250

        dataset = DeepSTARRDataset(
            targets_file=datafile,
            fasta_file=fastafile,
            tokenizer=tokenizer,
            max_seq_len=seq_len,
        )
        res = dataset.__getitem__(2)
        assert len(res["input_ids"]) == seq_len
        assert res["labels"].shape[0] == 2

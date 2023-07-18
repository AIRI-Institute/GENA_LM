import random

import numpy as np
from downstream_tasks.SpliceAI.SpliceAIDataset import SpliceAIDataset
from transformers import AutoTokenizer


class TestSpliceAIDataset:
    def sequence_tokenization(
        self, tokenizer, targets_offset, L_context_size, R_context_size, max_seq_len=512
    ):

        # create random sequence
        M_size = targets_offset

        left = random.choices(["A", "T", "G", "C"], k=L_context_size)
        right = random.choices(["A", "T", "G", "C"], k=R_context_size)
        mid = random.choices(["A", "T", "G", "C"], k=M_size)

        seq = (
            ["N"] * (max(0, targets_offset - len(left)))
            + left
            + mid
            + right
            + ["N"] * max(0, (targets_offset - len(right)))
        )
        seq = "".join(seq)

        # Add X to imitate any service token
        encoding = tokenizer(
            "".join(left + ["X"] + mid + ["X"] + right + ["X"]),
            add_special_tokens=False,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        targets, token_targets, mid_token_targets = (
            [],
            [[-100, -100]],
            [[-100, -100]],
        )  # Start with CLS token

        within_target = False
        target_probs = [0] * 98 + [
            1,
            2,
        ]  # ~1% probability for class 1 or 2, 98% of class 0
        for ind, val in enumerate(encoding["offset_mapping"][0]):
            if encoding["input_ids"][0][ind] == 0:  # reach "X" (or ["UNK"]) token
                token_targets.append([-100, -100])  # add SEP token labels
                mid_token_targets.append([-100, -100])  # add SEP token labels
                within_target = not within_target
                continue
            if within_target:
                token_length = val[1] - val[0]
                current_token_bp_targets = random.choices(target_probs, k=token_length)
                targets += current_token_bp_targets
                current_token_targets = [
                    1 if 1 in current_token_bp_targets else 0,
                    2 if 2 in current_token_bp_targets else 0,
                ]
                token_targets.append(current_token_targets)
                mid_token_targets.append(current_token_targets)
            else:
                token_targets.append([-100, -100])  # service token

        assert len(targets) == M_size

        if len(token_targets) < max_seq_len:
            token_targets.extend(
                [[-100, -100]] * (max_seq_len - len(token_targets))
            )  # imitate padding
        elif len(mid_token_targets) > max_seq_len:
            token_targets = np.concatenate(
                [mid_token_targets[: max_seq_len - 2], [[-100, -100]] * 2]
            )

        dataset_results = self.dataset.tokenize_inputs(seq, np.array(targets))
        a = np.array(token_targets)
        b = dataset_results["labels"]

        assert b.shape[0] == max_seq_len
        assert a.shape[1] == b.shape[1] == 2
        
        # we expect that a and b are either the same (if there was no trim)
        # or a contains b (if there was trim)
        if len(a)>len(b):
            b = b[1:-1] # remove first and last token (CLS and SEP)

        # now we check that returned token targets are substring of the tokens targets computed here
        substr = [
            x
            for x in range(len(a) - len(b) + 1)
            if np.all(np.equal(a[x : x + len(b)], b).flatten())
        ]
        assert len(substr) > 0

    def test_sequence_tokenization(self):
        targets_offset = 5000
        random.seed(42)

        tokenizer = AutoTokenizer.from_pretrained("data/tokenizers/t2t_1000h_multi_32k/",
                                                    )
        datafile = "downstream_tasks/SpliceAI/test_Dataset_data.csv"

        for max_seq_len in range(12, 1000, 61):
            self.dataset = SpliceAIDataset(
                datafile,
                tokenizer,
                max_seq_len=max_seq_len,
                targets_offset=targets_offset,
            )

            # test very small context
            for i in range(5):
                L_context_size = random.randint(0, targets_offset // 10)
                R_context_size = random.randint(0, targets_offset // 10)
                self.sequence_tokenization(
                    tokenizer,
                    targets_offset,
                    L_context_size,
                    R_context_size,
                    max_seq_len,
                )

            # test large context
            for i in range(5):
                L_context_size = random.randint(targets_offset - 10, targets_offset)
                R_context_size = random.randint(targets_offset - 10, targets_offset)
                self.sequence_tokenization(
                    tokenizer,
                    targets_offset,
                    L_context_size,
                    R_context_size,
                    max_seq_len,
                )

            # test random context
            for i in range(5):
                L_context_size = random.randint(0, targets_offset)
                R_context_size = random.randint(0, targets_offset)
                self.sequence_tokenization(
                    tokenizer,
                    targets_offset,
                    L_context_size,
                    R_context_size,
                    max_seq_len,
                )

    def longrun_tokens_class_computation(self):
        datafile = "test_Dataset_data.gz.df.pkl"
        tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base")
        targets_offset = 5000
        max_seq_len = 512

        self.dataset = SpliceAIDataset(
            datafile, tokenizer, max_seq_len=max_seq_len, targets_offset=targets_offset
        )

        dataset_results = self.dataset.__getitem__(0)

        seq, targets = self.dataset.data.iloc[0].values

        seq_encoding = tokenizer(
            seq,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_seq_len,
            return_offsets_mapping=True,
            return_tensors="np",
        )

        token_targets_class1 = []
        token_targets_class2 = []

        for om in seq_encoding["offset_mapping"][0]:
            st, en = om
            if en == 0 or st < targets_offset or st > targets_offset + len(targets):
                token_targets_class1.append(-100)
                token_targets_class2.append(-100)
            elif targets_offset + len(targets) >= st >= targets_offset:
                token_targets = np.unique(
                    targets[st - targets_offset : en - targets_offset]
                )
                if 1 in token_targets.tolist():
                    token_targets_class1.append(1)
                else:
                    token_targets_class1.append(0)
                if 2 in token_targets.tolist():
                    token_targets_class2.append(2)
                else:
                    token_targets_class2.append(0)
            else:
                raise ValueError

        for pos, (q, v) in enumerate(
            zip(token_targets_class1, dataset_results["labels"][:, 0])
        ):
            assert q == v, print("Class mismatch at pos ", pos, ":", q, v)

        for pos, (q, v) in enumerate(
            zip(token_targets_class2, dataset_results["labels"][:, 1])
        ):
            assert q == v, print("Class mismatch at pos ", pos, ":", q, v)
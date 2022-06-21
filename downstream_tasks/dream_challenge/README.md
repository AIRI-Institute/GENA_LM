# DREAM challenge
leaderboard: https://share.streamlit.io/muntakimrafi/dream-2022-random-promoter-challenge-leaderboard/main/src/app.py

Train and test sets could be downloaded from https://www.synapse.org/#!Synapse:syn28469146/wiki/617075:
- train_sequences.txt
- test_sequences.txt

## Data sample
```
TGCATTTTTTTCACATCTATGTTGCGTTAGAACGATATTGGAACACTTGTCAACAAGCTCATCTGAACTAATAGAGATGTATTCATAGGCTTCAGGTGGTTACGGCTGTT  6.0
TGCATTTTTTTCACATCGAGCTCACGCACTTGTCTACGAAAGGTCAAGGGCACTGGGTTTGGATGGCTATCGTGGCGCTACGACGGCTTTTTTCCTTGGTTACGGCTGTT  10.5490892577603
TGCATTTTTTTCACATCGTAGATAGATGGCACGGCGCCGTTACCTTTAAGGAAAATGTGGATCGGATGCAGAATTACTGAGGAATGCGGTTGAACACGGTTACGGCTGTT  12.0
TGCATTTTTTTCACATCTTTGTCCTCATTTGCCCCAGGCCCGAAGGATTATGGATTCTTAGATTCATTTGGTAAATAGATTCCTTAGTGTCCGGCTCGGTTACGGCTGTT  14.0
TGCATTTTTTTCACATCACGTGCTTGTTGGGCCCGGATATATACCATCAACTAATCGCGTATCATGTCGAACGTCCGCTATCTTTGCGCTTGTATAAGGTTACGGCTGTT  14.0
```
Target values in train set mean: 11.147008459472662, std: 2.3707070873733525

## Cross-validation
```bash
python build_folds.py --data_path ./train_sequences.txt --n 5
```

Cross-validation metrics are obtained by averaging scores on cross-val test sets:
``` 
split_1: train: [fold_1, fold_2, fold_3], valid: [fold_4], test: [fold_5]
split_2: train: [fold_2, fold_3, fold_4], valid: [fold_5], test: [fold_1]
...
split_5: train: [fold_5, fold_1, fold_2], valid: [fold_3], test: [fold_4]
```

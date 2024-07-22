# Predict sex by genome
This task is designed to test the hypothesis that a model can be trained to identify gender-related features across an entire human genome.
For human 57M bp (~2%) are in chrY.

Current approach: sample N random chunks from the genome uniformly and guess the sex. Not all samples will have chunks from chrY.

## Prepare data
Train/valid/test split of genomes:
```
python split_dataset.py --data_path /mnt/20tb/vsfishman/nn_interpretator/1000g_fasta_dataset/dataset/done \
    --labels_path /mnt/20tb/vsfishman/nn_interpretator/1000g_fasta_dataset/dataset/done/samples_done.txt \
    --save_folder /mnt/20tb/ykuratov/gender_data/
```

convert to hdf5:
```
python convert_to_hdf5.py --data_path /mnt/20tb/vsfishman/nn_interpretator/1000g_fasta_dataset/dataset/done \
  --train_csv /mnt/20tb/ykuratov/gender_data/train.csv \
  --valid_csv /mnt/20tb/ykuratov/gender_data/valid.csv \
  --test_csv /mnt/20tb/ykuratov/gender_data/test.csv \
  --save_folder /mnt/20tb/ykuratov/gender_data/
```


## Run model
```
CUDA_VISIBLE_DEVICES=0,1 NP=2 ./run_finetuning.sh
```
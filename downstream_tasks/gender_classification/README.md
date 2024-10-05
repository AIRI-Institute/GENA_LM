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


## Prepare mouse data
```bash
python split_dataset.py --data_path /mnt/20tb/vsfishman/nn_interpretator/mouse_fasta_dataset/dataset \
--labels_path /mnt/20tb/vsfishman/nn_interpretator/mouse_fasta_dataset/mouse_lines_metadata.txt \
--save_folder /mnt/20tb/ykuratov/mouse_gender_data/ \
--train_size 16 --valid_size 10 --test_size 10 --label_column gender
```
output:
```
size of train: 16 / 36 = 0.444
size of valid: 10 / 36 = 0.278
size of test: 10 / 36 = 0.278
```

```bash
python convert_to_hdf5.py --data_path /mnt/20tb/vsfishman/nn_interpretator/mouse_fasta_dataset/dataset \
--train_csv /mnt/20tb/ykuratov/mouse_gender_data/train.csv \
--valid_csv /mnt/20tb/ykuratov/mouse_gender_data/valid.csv \
--test_csv /mnt/20tb/ykuratov/mouse_gender_data/test.csv \
--save_folder /mnt/20tb/ykuratov/mouse_gender_data/ \
--sample_id_column strain_name
```
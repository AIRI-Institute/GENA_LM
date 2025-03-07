import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd


def main(data_path, labels_path, save_folder, label_column='sex', train_size=35, valid_size=10, test_size=24,
         random_seed=142):
    data_path = Path(data_path)
    save_folder = Path(save_folder)

    df = pd.read_csv(labels_path, sep='\t')

    total_size = train_size + valid_size + test_size

    if total_size > len(df):
        raise ValueError("Total size of train, valid, and test sets exceeds the number of samples in the DataFrame.")

    train_ratio = train_size / total_size
    valid_ratio = valid_size / total_size
    test_ratio = test_size / total_size


    train_df, remaining_df = train_test_split(df, train_size=train_ratio, stratify=df[label_column],
                                              random_state=random_seed)

    valid_test_ratio = valid_ratio / (valid_ratio + test_ratio)

    valid_df, test_df = train_test_split(remaining_df, train_size=valid_test_ratio, stratify=remaining_df[label_column],
                                         random_state=random_seed)

    data_splits = {
        'train': train_df,
        'valid': valid_df,
        'test': test_df,
    }

    for split_name in data_splits:
        split_size = len(data_splits[split_name])
        print(f'size of {split_name}: {split_size} / {total_size} = {split_size/total_size:.3f}')

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    for split_name in data_splits:
        # save targets
        data_splits[split_name].reset_index(drop=True).to_csv(save_folder / f'{split_name}.csv')


"""
python split_dataset.py --data_path /mnt/20tb/vsfishman/nn_interpretator/1000g_fasta_dataset/dataset/done \
    --labels_path /mnt/20tb/vsfishman/nn_interpretator/1000g_fasta_dataset/dataset/done/samples_done.txt \
    --save_folder /mnt/20tb/ykuratov/gender_data/
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, validation, and test sets")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the csv with labels')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save the splits')
    parser.add_argument('--label_column', type=str, default='sex', help='Column name for labels')
    parser.add_argument('--train_size', type=int, default=35, help='Number of samples in the training set')
    parser.add_argument('--valid_size', type=int, default=10, help='Number of samples in the validation set')
    parser.add_argument('--test_size', type=int, default=24, help='Number of samples in the test set')
    parser.add_argument('--random_seed', type=int, default=142, help='Random seed for reproducibility')

    args = parser.parse_args()

    main(args.data_path, args.labels_path, args.save_folder, args.label_column,
         args.train_size, args.valid_size, args.test_size,
         args.random_seed)

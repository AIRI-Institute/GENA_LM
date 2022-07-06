import argparse
import pandas as pd
import random
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='./dataset_train_all.csv.gz', help='path to the train dataset')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='how many valid samples to take (default: 0.1)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    args.data_path = Path(args.data_path)
    print('reading data..')
    data = pd.read_csv(args.data_path, sep=',', header=None)
    valid_size = int(len(data) * args.valid_ratio)
    print('shuffling indices..')
    indices = list(range(len(data)))
    random.shuffle(indices)
    print('building train/val split..')
    valid_data = data.iloc[indices[:valid_size], :]
    train_data = data.iloc[indices[valid_size:], :]
    print('saving data..')
    valid_data.to_csv(args.data_path.parent / 'valid.csv.gz', sep=',', index=False, header=None, compression='gzip')
    train_data.to_csv(args.data_path.parent / 'train.csv.gz', sep=',', index=False, header=None, compression='gzip')
    print('done')

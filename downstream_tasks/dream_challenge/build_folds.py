import argparse
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='./train_sequences.txt', help='path to the dataset')
parser.add_argument('--n', type=int, default=5, help='number of splits (default: 5)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

if __name__ == '__main__':
    args = parser.parse_args()

    data = pd.read_csv(args.data_path, sep='\t', header=None)
    kf = KFold(n_splits=args.n, shuffle=True, random_state=args.seed)
    for i, (_, test) in tqdm(enumerate(kf.split(data)), total=args.n):
        data.iloc[test, :].to_csv(f'./fold_{i+1}.txt', sep='\t', header=False, index=False)

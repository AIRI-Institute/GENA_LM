import argparse
from pathlib import Path

import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='train.mat', help='path to deepsea train data in mat format')
parser.add_argument('--valid_path', type=str, default='valid.mat', help='path to deepsea valid data in mat format')
parser.add_argument('--test_path', type=str, default='test.mat', help='path to deepsea test data in mat format')


def convert_sequence(encoded_seq):
    BASES_ARR = np.array(['N', 'A', 'G', 'C', 'T'])
    # check that encoding is full of zeros or exactly one-hot
    assert np.all(encoded_seq.sum(axis=0) <= 1.0)
    # if all zeros we choose N, otherwise one of AGCT will be chosen
    encoded_seq = np.concatenate([zeros, encoded_seq], axis=0).argmax(axis=0)
    return ''.join(BASES_ARR[encoded_seq])


if __name__ == '__main__':
    args = parser.parse_args()
    print('preparing valid/test data..')
    zeros = np.zeros((1, 1000))
    for path, datasplit in tqdm(zip([args.valid_path, args.test_path], ['valid', 'test']), total=2):
        path = Path(path)
        data = scipy.io.loadmat(path)
        d = []
        for i in tqdm(range(len(data[f'{datasplit}xdata']))):
            seq = convert_sequence(data[f'{datasplit}xdata'][i])
            targets = data[f'{datasplit}data'][i]
            d += [[seq] + targets.tolist()]
        print('creating dataframe..')
        df = pd.DataFrame(d)
        print('saving..')
        df.to_csv(path.parent / f'{datasplit}.csv.gz', sep=',', index=False, header=False, compression='gzip')
        del d, df

    print('preparing train data..')
    args.train_path = Path(args.train_path)
    f = h5py.File(args.train_path, 'r')
    train_data = {k: np.array(f.get(k)) for k in f.keys()}
    train_data['trainxdata'] = train_data['trainxdata'].transpose(2, 1, 0)
    train_data['traindata'] = train_data['traindata'].transpose(1, 0)
    datasplit = 'train'
    data = train_data
    d = []
    for i in tqdm(range(len(data[f'{datasplit}xdata']))):
        seq = convert_sequence(data[f'{datasplit}xdata'][i])
        targets = data[f'{datasplit}data'][i]
        d += [[seq] + targets.tolist()]
    print('creating dataframe..')
    df = pd.DataFrame(d)
    print('saving..')
    df.to_csv(args.train_path.parent / f'{datasplit}.csv.gz', sep=',', index=False, header=False, compression='gzip')
    print('done')

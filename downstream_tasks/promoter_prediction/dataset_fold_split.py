import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedKFold

print('Please, input your dataset as csv file: ')
filename = input()
df = pd.read_csv(filename)
ind = filename.find('len')
name = filename[ind:].split('_')[0] + '_' + filename[ind:].split('_')[1]
if not os.path.exists(f'{name}_fold_split'):
    os.mkdir(f'{name}_fold_split')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits)
target = df.loc[:, 'promoter_presence']

print('writing folds...')
for fold_no, (_, val_index) in enumerate(skf.split(df, target)):
    val = df.loc[val_index, :]
    val.to_csv(f'{name}_fold_split/fold_{fold_no+1}.csv', index=False)

print('writing train/valid/test splits...')
for i in range(n_splits):
    fold_ind = (np.arange(0, n_splits) + i) % n_splits + 1
    train_folds = fold_ind[:-2]
    valid_fold = fold_ind[-2]
    test_fold = fold_ind[-1]
    print(train_folds, valid_fold, test_fold)
    os.makedirs(os.path.dirname(f'{name}_fold_split/split_{i+1}/train/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'{name}_fold_split/split_{i+1}/valid/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'{name}_fold_split/split_{i+1}/test/'), exist_ok=True)
    for ind in train_folds:
        shutil.copy(f'{name}_fold_split/fold_{ind}.csv', f'{name}_fold_split/split_{i+1}/train/')
    shutil.copy(f'{name}_fold_split/fold_{valid_fold}.csv', f'{name}_fold_split/split_{i+1}/valid/')
    shutil.copy(f'{name}_fold_split/fold_{test_fold}.csv', f'{name}_fold_split/split_{i+1}/test/')

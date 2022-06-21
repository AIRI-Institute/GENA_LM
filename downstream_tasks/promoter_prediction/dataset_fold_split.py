import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold

print('Please, input your dataset as csv file: ')
filename = input()
df = pd.read_csv(filename)
ind = filename.find('len')
name = filename[ind:].split('_')[0] + '_' + filename[ind:].split('_')[1]
if not os.path.exists(f'{name}_fold_split'):
    os.mkdir(f'{name}_fold_split')

skf = StratifiedKFold(n_splits=5)
target = df.loc[:,'promoter_presence']

fold_no = 1
for train_index, val_index in skf.split(df, target):
    val = df.loc[val_index,:]
    val.to_csv(f'{name}_fold_split/fold_' + str(fold_no) + '.csv', index=False)
    fold_no += 1

"""
@author: Hongzuo Xu
@comments: testbed for time series anomaly detection on synthetic data
(An experiment on the generalization ability to different types of time series anomalies)
"""

import os
import numpy as np
import pandas as pd
import argparse
from src.algorithms.couta_algo import COUTA


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default=f'point')
args = parser.parse_args()

# read data
df = pd.read_csv(f'data_processed/synthetic_{args.type}.csv', index_col=0)
split = int(0.4*df.shape[0])
train_df = df.iloc[: split]
test_df = df.iloc[split:]
label01 = df['label01'].values[split:]
labelmul = df['labelmul'].values[split:]
df = df.drop(['label01', 'labelmul'], axis=1)

res_dir = '@results_showcase/'
os.makedirs(res_dir, exist_ok=True)

model_configs = {
    'sequence_length': 50,
    'stride': 1,
    'lr': 0.001,
    'num_epochs': 40,
    'kernel_size': 2,
    'hidden_dims': [16],
    'emb_dim': 16,
    'rep_hidden': 64,
    'dropout': 0.0,
    'alpha': 0.1,
    'bias': 1,
    'neg_batch_ratio': 0.2,
    'train_val_pc': 0,
    'es': 0,
    'seed': 42,
    'ss_type': 'full'
}
model = COUTA(**model_configs)
model.fit(train_df)
scores = model.predict(test_df)['score_t']
np.save(res_dir + f'showcase_{args.type}_score_couta.npy', scores)



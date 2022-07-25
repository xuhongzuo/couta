import pickle
import os
import numpy as np
import pandas as pd
from src.experiments.utils import split_data, shuffle_data, adjust_contamination


data_name = 'DSADS'
save_path = '../../data_processed_contam/'
p_lst = ['p' + str(i) for i in range(1,9)]
seed = 42
ratios = [0.0, 0.04, 0.08, 0.12, 0.16]


# read data from disk
path = '../../data/DASADS/'
all_data = pickle.load(open(path + 'all_data.pkl', 'rb'))
all_label = pickle.load(open(path + 'all_label.pkl', 'rb'))
dim = all_data['p1'].shape[-1]
seq_len = all_data['p1'].shape[1]


# get the 0/1 label
abnormal_class = ['a12', 'a17', 'a18']
all_label2 = {}
for p in p_lst:
    label = all_label[p]
    new_label = np.zeros(len(label))
    for i in range(len(label)):
        if label[i] in abnormal_class:
            new_label[i] = 1
    all_label2[p] = new_label



for p in p_lst:
    print(p)
    label = all_label2[p]
    data = all_data[p]
    data, label = shuffle_data(data, label, seed)

    # keep the same anomaly ratio in training/testing set
    train_normal, train_anomaly, test_normal, test_anomaly = split_data(data, label, ratio=0.6)

    print(len(train_anomaly) / (len(train_anomaly) + len(train_normal)))

    # randomly remove contaminated anomalies in training set
    for target_ratio in ratios:

        df_train, df_test = adjust_contamination(train_normal, train_anomaly, test_normal, test_anomaly,
                                                 target_ratio, seed=seed)


        print(df_train.shape, df_train['label'].sum(), df_test.shape, df_test['label'].sum())

        dir = save_path + f'{data_name}_contam_{target_ratio}/{p}_seed{seed}/'
        os.makedirs(dir, exist_ok=True)
        df_train.to_csv(dir + f'{p}_seed{seed}_train.csv')
        df_test.to_csv(dir + f'{p}_seed{seed}_test.csv')

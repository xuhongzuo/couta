import os
import numpy as np
from src.experiments.utils import shuffle_data, split_data, adjust_contamination



# read data from disk
data_name = 'EP'
path = '../../data/epilepsy/'
save_path = '../../data_processed_contam/'
seed = 42
ratios = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24]


train_x = np.load(path + 'train_array.npy')
train_y = np.load(path + 'train_label.npy')
test_x = np.load(path + 'test_array.npy')
test_y = np.load(path + 'test_label.npy')
data = np.concatenate([train_x, test_x])
label = np.hstack([train_y, test_y])
dim = data.shape[2]
seq_len = data.shape[1]

print(data.shape)


# get the 0/1 label
abnormal_class = ['EPILEPSY']
new_label = np.zeros(len(label))
for i in range(len(label)):
    if label[i] in abnormal_class:
        new_label[i] = 1
label = new_label


data, label = shuffle_data(data, label, seed)

# keep the same anomaly ratio in training/testing set
train_normal, train_anomaly, test_normal, test_anomaly = split_data(data, label, ratio=0.6)
print('original train shape', train_normal.shape, train_anomaly.shape)
print('original test shape', test_normal.shape, test_anomaly.shape)

print(len(train_anomaly) / (len(train_anomaly) + len(train_normal)))

# randomly remove contaminated anomalies in training set
for target_ratio in ratios:

    for i in range(5):
        s = seed+i
        df_train, df_test = adjust_contamination(train_normal, train_anomaly,
                                                 test_normal, test_anomaly,
                                                 target_ratio, seed=s)

        print(df_train.shape, df_train['label'].sum(), df_test.shape, df_test['label'].sum())


        dir = save_path + f'{data_name}_contam_{target_ratio}/seed{s}/'
        os.makedirs(dir, exist_ok=True)
        df_train.to_csv(dir + f'seed{s}_train.csv')
        df_test.to_csv(dir + f'seed{s}_test.csv')

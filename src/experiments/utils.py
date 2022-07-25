# -*- coding: utf-8 -*-
# @Time    : 2022/5/22
# @Author  : Xu Hongzuo
# @File    : utils.py
# @Comment :
import pandas as pd
import numpy as np
from collections import Counter


def shuffle_data(data, label, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(data))
    data = data[idx]
    label = label[idx]
    return data, label


def reshape_data(x, y):
    # reshape
    _dim = x.shape[2]
    _seq_len = x.shape[1]
    x2 = x.reshape(-1, _dim)
    y2 = y.repeat(_seq_len)
    return x2, y2


def split_data(data, label, ratio=0.6):
    # keep the same anomaly ratio in training/testing set
    normal_data = data[np.where(label == 0)[0]]
    anomaly_data = data[np.where(label == 1)[0]]

    split1 = int(ratio * len(normal_data))
    split2 = int(ratio * len(anomaly_data))

    train_normal = normal_data[:split1]
    train_anomaly = anomaly_data[:split2]
    test_normal = normal_data[split1:]
    test_anomaly = anomaly_data[split2:]

    return train_normal, train_anomaly, test_normal, test_anomaly


def adjust_contamination(train_normal, train_anomaly, test_normal, test_anomaly, target_ratio, seed):
    print('\noriginal contam ratio:', len(train_anomaly)/ (len(train_normal)+len(train_anomaly)))

    contam_num = int((target_ratio / (1. - target_ratio)) * len(train_normal))
    print('target ratio and contam num:', target_ratio, contam_num)

    rng = np.random.RandomState(seed)
    # randomly under-sampling
    if contam_num < len(train_anomaly):
        idx = rng.choice(len(train_anomaly), contam_num, replace=False)

    # randomly over-sampling, keep the a full set of anomalies
    else:
        idx1 = np.arange(len(train_anomaly))
        idx2 = rng.choice(len(train_anomaly), contam_num - len(train_anomaly), replace=True)
        idx = np.hstack([idx1, idx2])

    print('selected idx', sorted(idx))
    train_anomaly2 = train_anomaly[idx]

    train_data = np.vstack([train_normal, train_anomaly2])
    train_y = np.hstack([np.zeros(len(train_normal), dtype=int), np.ones(len(train_anomaly2), dtype=int)])
    print('train normal/anomaly shape', train_data.shape, Counter(train_y))

    # shuffle and reshape training data
    train_data, train_y = shuffle_data(train_data, train_y, seed)
    train_data, train_y = reshape_data(train_data, train_y)


    # move removed anomaly to testing set
    idx2 = np.ones(len(train_anomaly), np.bool)
    idx2[idx] = False
    removed_train_anomaly = train_anomaly[idx2]
    print('removed shape', removed_train_anomaly.shape)

    test_data = np.vstack([test_normal, test_anomaly, removed_train_anomaly])
    test_y = np.hstack([np.zeros(len(test_normal), dtype=int),
                        np.ones(len(test_anomaly), dtype=int),
                        np.ones(len(removed_train_anomaly), dtype=int)])
    print('test normal/anomaly shape', test_data.shape, Counter(test_y))

    # test_data = np.vstack([test_normal, test_anomaly])
    # test_y = np.hstack([np.zeros(len(test_normal), dtype=int), np.ones(len(test_anomaly), dtype=int)])


    # shuffle and reshape testing data
    test_data, test_y = shuffle_data(test_data, test_y, seed)
    test_data, test_y = reshape_data(test_data, test_y)

    dim = train_data.shape[1]
    df_train = pd.DataFrame(train_data, columns=['A' + str(i) for i in range(dim)])
    df_train['label'] = train_y

    df_test = pd.DataFrame(test_data, columns=['A' + str(i) for i in range(dim)])
    df_test['label'] = test_y
    return df_train, df_test
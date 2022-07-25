import sys
sys.path.append('../src')
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import seaborn as sns
from src import utils_eval


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_sequences_data(data, data_root):
    name_lst = []
    train_df_lst = []
    test_df_lst = []
    label_lst = []

    train_x = np.load(f'{data_root}{data}/{data}_trainx.npy')
    test_x = np.load(f'{data_root}{data}/{data}_testx.npy')
    train_y = np.load(f'{data_root}{data}/{data}_trainy.npy')
    test_y = np.load(f'{data_root}{data}/{data}_testy.npy')

    print(train_x.shape, test_x.shape)

    train_df_lst.append(train_x)
    test_df_lst.append(test_x)
    label_lst.append(test_y)
    name_lst.append(data)

    return train_df_lst, test_df_lst, label_lst, name_lst


def minmax_norm(arr):
    arr = np.array(arr)
    _min_, _max_ = np.min(arr), np.max(arr)
    r = _max_ - _min_ if _max_ != _min_ else 1
    arr_new = np.array([(a - _min_) / r for a in arr])

    return arr_new


def data_standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = X_train.min(), X_train.max()
    for col in X_train.columns:
        if maxi[col] != mini[col]:
            X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
        else:
            assert X_train[col].nunique() == 1
            if remove:
                if verbose:
                    print("Column {} has the same min and max value in train. Will remove this column".format(col))
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
            else:
                if verbose:
                    print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                if mini[col] != 0:
                    X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                    X_test[col] = X_test[col] / mini[col]
                if verbose:
                    print("After transformation, train unique vals: {}, test unique vals: {}".format(
                    X_train[col].unique(),
                    X_test[col].unique()))
    return X_train, X_test


def meta_process_scores(predictions_dic, name):
    if predictions_dic["score_t"] is None:
        assert predictions_dic['error_tc'] is not None
        predictions_dic['score_tc'] = predictions_dic['error_tc']
        predictions_dic['score_t'] = np.sum(predictions_dic['error_tc'], axis=1)

    """
    Following [Garg 2021 TNNLS], Unlike the other datasets, each entity in MSL and SMAP consists of only 1 sensor
    while all the other channels are one-hot-encoded commands given to that entity. 
    Therefore, for dataset MSL and SMAP, use all channels as input to the models, but use the model error of only 
    the sensor channel for anomaly detection.
    """
    if 'MSL' in name or 'SMAP' in name:
        # if error_tc is not None:
        #     predictions_dic['error_t'] = error_tc[:, 0]
        if predictions_dic['score_tc'] is not None:
            predictions_dic['score_t'] = predictions_dic['score_tc'][:, 0]

    return predictions_dic


# ----------------------------------------- visualization --------------------------------------- #

def plt_full_chart(df, y=None, save_path=None):
    n_dim = df.shape[1]
    length = df.shape[0]

    fig = plt.figure(figsize=(20, 1.8*n_dim))

    anom_pairs = []
    if y is not None:
        anom_index = np.where(y==1)[0]
        tmp_seg = []
        for i in anom_index:
            tmp_seg.append(i)
            if i + 1 not in anom_index:
                anom_pairs.append((tmp_seg[0], tmp_seg[-1]))
                tmp_seg = []

    palette = sns.color_palette("cividis", n_dim)
    sns.set_theme(palette=palette, style='ticks')

    for ii in range(n_dim):
        value = df.iloc[:, ii].values
        col = df.columns[ii]
        fig.add_subplot(n_dim, 1, ii + 1)
        sns.lineplot(x=np.arange(length), y=value, legend=True, label=col, color=palette[0])

        value_min = np.around(value.min(), decimals=1) - 0.1
        value_max = np.around(value.max(), decimals=1) + 0.1

        y = np.linspace(value_min, value_max, 10)
        for pair in anom_pairs:
            x1 = np.ones(len(y)) * pair[0]
            x2 = np.ones(len(y)) * pair[1]
            plt.fill_betweenx(y, x1, x2, alpha=0.3, color='goldenrod')

        plt.legend(loc='upper right')
        plt.xlim(0, length)
        plt.ylim(value_min, value_max)
        if ii != n_dim-1:
            plt.xticks([])
            plt.xlabel('')

    plt.subplots_adjust(wspace=0, hspace=0.09)

    if save_path is not None:
        plt.savefig(save_path)

    return

def plt_res_with_dat(score_lst, y, data, name_lst=None, ytick_range=None):

    n_algo = len(score_lst)
    fig = plt.figure(figsize=(15, 4*n_algo))
    index = np.arange(len(score_lst[0]))

    anom_pairs = []
    if y is not None:
        anom_index = np.where(y == 1)[0]
        tmp_seg = []
        for i in anom_index:
            tmp_seg.append(i)
            if i + 1 not in anom_index:
                anom_pairs.append((tmp_seg[0], tmp_seg[-1]))
                tmp_seg = []



    fig.add_subplot(n_algo+1, 1, 1)
    sns.lineplot(x=index, y=data, color=sns.color_palette('Greys_r')[0])
    plt.xlim(index[0], index[-1])
    value_min = np.around(data.min(), decimals=1) - 0.1
    value_max = np.around(data.max(), decimals=1) + 0.1
    values_ = np.linspace(value_min, value_max, 10)
    for pair in anom_pairs:
        x1 = np.ones(len(values_)) * pair[0]
        x2 = np.ones(len(values_)) * pair[1]
        plt.fill_betweenx(values_, x1, x2, alpha=0.3, color='goldenrod')

    for ii, score in enumerate(score_lst):
        fig.add_subplot(n_algo+1, 1, 2+ii)
        palette = sns.color_palette("cividis")
        sns.set_theme(palette=palette, style='ticks')

        # # scale scores
        # if np.max(score)!= np.min(score):
        #     score = (score - np.min(score)) / (np.max(score) - np.min(score))
        # else:
        #     score = np.zeros_like(score)
        adj_score = utils_eval.adjust_scores(y, score)

        sns.lineplot(x=index, y=score, legend=True, color=palette[0])

        value_min = np.around(score.min(), decimals=1) - 0.1
        value_max = np.around(score.max(), decimals=1) + 0.1
        values_ = np.linspace(value_min, value_max, 10)
        for pair in anom_pairs:
            x1 = np.ones(len(values_)) * pair[0]
            x2 = np.ones(len(values_)) * pair[1]
            plt.fill_betweenx(values_, x1, x2, alpha=0.3, color='goldenrod')

        plt.xlim(index[0], index[-1])
        plt.ylim(0, 1)
        if ytick_range is not None:
            plt.ylim(ytick_range[0], ytick_range[1])

        if name_lst is not None:
            plt.title(name_lst[ii])


    plt.show()
    return fig



def plt_res(score, y, ytick_range=None):
    fig = plt.figure(figsize=(20, 2))

    adj_score = utils_eval.adjust_scores(y, score)
    best_f1, best_p, best_r, best_th = utils_eval.get_best_f1(y, adj_score)

    plt.axhline(best_th, color='r', linewidth=0.4, linestyle='-.')

    anom_pairs = []
    if y is not None:
        anom_index = np.where(y == 1)[0]
        tmp_seg = []
        for i in anom_index:
            tmp_seg.append(i)
            if i + 1 not in anom_index:
                anom_pairs.append((tmp_seg[0], tmp_seg[-1]))
                tmp_seg = []

    palette = sns.color_palette("cividis")
    sns.set_theme(palette=palette, style='ticks')

    index = np.arange(len(score))
    # sns.scatterplot(x=index, y=score, legend=True, color=palette[0], s=5, marker='x')
    sns.lineplot(x=index, y=score, legend=True, color=palette[0])
    value_min = np.around(score.min(), decimals=1) - 0.1
    value_max = np.around(score.max(), decimals=1) + 0.1

    values_ = np.linspace(value_min, value_max, 10)
    for pair in anom_pairs:
        x1 = np.ones(len(values_)) * pair[0]
        x2 = np.ones(len(values_)) * pair[1]
        plt.fill_betweenx(values_, x1, x2, alpha=0.3, color='goldenrod')

    plt.xlim(index[0], index[-1])
    plt.ylim(value_min, value_max)
    if ytick_range is not None:
        plt.ylim(ytick_range[0], ytick_range[1])
    plt.show()
    return fig


def plt_res_multi(score_lst, y, title_lst=None, ytick_range=None):
    n_algo = len(score_lst)
    fig = plt.figure(figsize=(20, 1.6 * n_algo))

    for aa in range(n_algo):
        ax = fig.add_subplot(n_algo, 1, aa + 1)

        score = score_lst[aa]
        adj_score = utils_eval.adjust_scores(y, score)
        best_f1, best_p, best_r, best_th = utils_eval.get_best_f1(y, adj_score)

        plt.axhline(best_th, color='r', linewidth=0.4, linestyle='-.')

        anom_pairs = []
        if y is not None:
            anom_index = np.where(y == 1)[0]
            tmp_seg = []
            for i in anom_index:
                tmp_seg.append(i)
                if i + 1 not in anom_index:
                    anom_pairs.append((tmp_seg[0], tmp_seg[-1]))
                    tmp_seg = []

        palette = sns.color_palette("cividis")
        sns.set_theme(palette=palette, style='ticks')

        index = np.arange(len(score))
        # sns.scatterplot(x=index, y=score, legend=True, color=palette[0], s=5, marker='x')
        sns.lineplot(x=index, y=score, legend=True, label=title_lst[aa], color=palette[0])
        value_min = np.around(score.min(), decimals=1) - 0.1
        value_max = np.around(score.max(), decimals=1) + 0.1

        values_ = np.linspace(value_min, value_max, 10)
        for pair in anom_pairs:
            x1 = np.ones(len(values_)) * pair[0]
            x2 = np.ones(len(values_)) * pair[1]
            plt.fill_betweenx(values_, x1, x2, alpha=0.3, color='goldenrod')

        plt.xlim(index[0], index[-1])
        plt.ylim(value_min, value_max)
        if ytick_range is not None:
            plt.ylim(ytick_range[0], ytick_range[1])

        if aa != n_algo - 1:
            plt.xticks([])

    plt.subplots_adjust(wspace=0, hspace=0.1)
    return fig
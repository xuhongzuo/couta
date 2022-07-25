"""
@author: Hongzuo Xu
@comments: Utility functions
"""

import os
import time
import yaml
import glob
import pandas as pd
import logging.config
import random
import string
from src.utils_eval import get_metrics, adjust_scores
from src.utils_general import data_standardize, meta_process_scores, plt_res, minmax_norm

intermediate_dir = './z-intermediate_model_files/'
model_configs_dir = 'configs/'
logging_configs_path = 'configs/configs_logging.yaml'


def get_model_configs(algo, data):
    # this is for robustness experiment, datasets are named as DSADS_contam_xx or EP_contam_xx
    if 'DSADS_' in data: data =  'DSADS'
    if 'EP_' in data: data = 'Epilepsy'

    try:
        print('get configs from yaml')
        path = f'configs/COUTA.yaml'
        with open(path) as f:
            d = yaml.safe_load(f)
            try:
                model_configs = d[data]
            except KeyError:
                model_configs = d['ASD']

    except FileNotFoundError:
        print('warning: use default settings')
        model_configs = {}

    if 'DSADS' in data:
        model_configs['sequence_length'] = 125
        model_configs['stride'] = 125
    elif 'Epilepsy' in data:
        model_configs['sequence_length'] = 206
        model_configs['stride'] = 206

    # for those big datasets, use higher stride values
    elif 'SWaT' in data:
        model_configs['stride'] = 100
    elif 'WaQ' in data:
        model_configs['stride'] = 5

    print(model_configs)
    return model_configs


def get_logger(log_path, results_raw_metrics_path, results_avg_metrics_path):
    with open(logging_configs_path, "r") as f:
        dict_conf = yaml.safe_load(f)

    dict_conf['handlers']['fh']['filename'] = log_path
    dict_conf['handlers']['fh_avg']['filename'] = results_avg_metrics_path
    dict_conf['handlers']['fh_raw']['filename'] = results_raw_metrics_path

    logging.config.dictConfig(dict_conf)

    logger_fh = logging.getLogger('logger_fh')
    logger_fh_raw = logging.getLogger('logger_fh_raw')
    logger_fh_avg = logging.getLogger('logger_fh_avg')
    # logger_sh = logging.getLogger('logger_sh')
    return logger_fh, logger_fh_raw, logger_fh_avg


def get_data_lst(data, data_root, entities=None):
    # if entities == 'SLC':
    #     with open(data_configs_path, "r") as f:
    #         conf = yaml.safe_load(f)
    #     try:
    #         entities = conf['selected_entities'][data]
    #     except KeyError:
    #         entities = 'FULL'

    if type(entities) == str:
        entities_lst = entities.split(',')
    elif type(entities) == list:
        entities_lst = entities
    else:
        raise ValueError('wrong entities')

    name_lst = []
    train_df_lst = []
    test_df_lst = []
    label_lst = []

    if len(glob.glob(os.path.join(data_root, data) + '/*.csv')) == 0:
        machine_lst = os.listdir(data_root + data + '/')
        for m in sorted(machine_lst):
            if entities != 'FULL' and m not in entities_lst:
                continue
            train_path = glob.glob(os.path.join(data_root, data, m, '*train*.csv'))
            test_path = glob.glob(os.path.join(data_root, data, m, '*test*.csv'))

            assert len(train_path) == 1 and len(test_path) == 1
            train_path, test_path = train_path[0], test_path[0]

            train_df = pd.read_csv(train_path, sep=',', index_col=0)
            test_df = pd.read_csv(test_path, sep=',', index_col=0)
            labels = test_df['label'].values
            train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

            train_df_lst.append(train_df)
            test_df_lst.append(test_df)
            label_lst.append(labels)
            name_lst.append(m)

    else:
        train_df = pd.read_csv(f'{data_root}{data}/{data}_train.csv', sep=',', index_col=0)
        test_df = pd.read_csv(f'{data_root}{data}/{data}_test.csv', sep=',', index_col=0)
        labels = test_df['label'].values
        train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

        train_df_lst.append(train_df)
        test_df_lst.append(test_df)
        label_lst.append(labels)
        name_lst.append(data)

    return train_df_lst, test_df_lst, label_lst, name_lst


def prepare(args):
    res_root = args.results_dir

    cur_time = time.strftime("%Y-%m-%d %H.%M", time.localtime())
    cur_time2 = time.strftime("%m%d_%H.%M", time.localtime())
    mask = ''.join(random.sample(string.ascii_letters, 5))
    if not args.save_pred:
        results_raw_dir = os.path.join(res_root, f'raw/')
    else:
        results_raw_dir = os.path.join(res_root, f'raw/raw-record@{args.algo}_{args.data}_{cur_time}_#{mask}_{args.flag}/')

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(res_root, exist_ok=True)
    os.makedirs(os.path.join(res_root, 'raw/'), exist_ok=True)
    os.makedirs(os.path.join(res_root, 'report/'), exist_ok=True)
    os.makedirs(results_raw_dir, exist_ok=True)

    results_raw_metrics_path = os.path.join(results_raw_dir, f'@raw_{args.algo}{args.flag}-{args.data}.csv')
    results_avg_metrics_path = os.path.join(res_root, f'report/{args.algo}{args.flag}-{args.data}.csv')
    log_path = os.path.join(f'{args.log_path}', f'{args.algo}{args.flag}-{args.data}-{cur_time2}.log')

    logger = get_logger(log_path, results_raw_metrics_path, results_avg_metrics_path)
    # logger 0 is log file
    # logger 1 is raw file
    # logger 2 is results file

    # # get model class and model parameters
    if 'COUTA' in args.algo:
        from src.algorithms.couta_algo import COUTA
        model_class = COUTA
    else:
        from src.algorithms.canonical_oc_algo import Canonical
        model_class = Canonical

    model_configs = get_model_configs(args.algo, args.data)
    return results_raw_dir, model_class, model_configs, logger


def run(train_df, test_df, labels, od_model, data_name):
    """

    Parameters
    ----------
        train_df:
        test_df:
        labels:
        od_model:
        data_name:

    Returns
    ----------

    """
    train_df, test_df = data_standardize(train_df, test_df, remove=False)
    train_df, test_df = train_df.interpolate(), test_df.interpolate()
    train_df, test_df = train_df.bfill(), test_df.bfill()

    od_model.fit(train_df)
    prediction = od_model.predict(test_df)

    prediction = meta_process_scores(prediction, data_name)
    scores = prediction['score_t']

    eval_info = get_metrics(labels, scores)
    adj_eval_info = get_metrics(labels, adjust_scores(labels, scores))

    return prediction, eval_info, adj_eval_info



"""
@author: Hongzuo Xu
@comments: testbed for time series anomaly detection
"""
import argparse
import os
import time
import pickle
import numpy as np
from main_utils import prepare, run, get_data_lst


# -------------------------------- argument parser --------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=f'data_processed/')
parser.add_argument('--results_dir', type=str, default='@results/', help='dataset name')
parser.add_argument('--data', type=str, default='ASD',help='dataset name')
parser.add_argument("--entities", type=str, default='omi-1',
                    help='FULL represents all the entities, or a list of entity names split by comma')
parser.add_argument('--algo', type=str, default='COUTA',
                    choices=['COUTA', 'COUTA_wto_nac', 'COUTA_wto_umc', 'Canonical'])

parser.add_argument('--device', help='torch device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--runs', help='', type=int, default='1')
parser.add_argument('--log_path', type=str, default='log/')
parser.add_argument('--save_pred', action='store_true', default=False)
parser.add_argument('--flag', type=str, default='')
parser.add_argument('--record_avg', type=int, default=1)

args = parser.parse_args()


# -------------------------------- running preparation --------------------------------#
results_raw_dir, model_class, model_configs, logger = prepare(args)
logger_fh, logger_fh_raw, logger_fh_avg = logger

# print the header of results files
cur_time = time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())
header = f'\n' \
         f'--------------------------------------------------------------------\n' \
         f'Time: {cur_time}, flag: {args.flag} \n' \
         f'Data: {args.data}, Algo: {args.algo}, Runs: {args.runs} \n' \
         f'Configs: {model_configs} \n' \
         f'--------------------------------------------------------------------\n'
logger_fh.info(header)
logger_fh_raw.info(header)
logger_fh_avg.info(header)

header2 = f'data, adj_auroc, adj_aupr, adj_f1, adj_p, adj_r, ' \
          f'adj_auroc_std, adj_aupr_std, adj_f1_std, adj_p_std, adj_r_std, time'
logger_fh_avg.info(header2)

# -------------------------------- Reading Data --------------------------------#
train_df_lst, test_df_lst, label_lst, name_lst = get_data_lst(args.data, args.data_dir, entities=args.entities)
name_lst = [args.data + '-' + n for n in name_lst]


# -------------------------------- Running --------------------------------#
start_time = time.time()
f1_lst = []
aupr_lst = []
for train, test, label, name in zip(train_df_lst, test_df_lst, label_lst, name_lst):
    entries = []
    t_lst = []
    for i in range(args.runs):
        logger_fh.info(f'\n\n Running {args.algo} on {name} [{i+1}/{args.runs}], '
                       f'cur_time: {time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())}')

        t1 = time.time()

        # running
        model_configs['seed'] = 42 + i
        model_configs['umc'] = 0 if 'wto_umc' in args.algo else 1
        model_configs['nac'] = 0 if 'wto_nac' in args.algo else 1

        model = model_class(**model_configs)
        predictions, eval_metrics, adj_eval_metrics = run(train, test, label, model, data_name=name)
        entries.append(adj_eval_metrics)

        t = time.time() - t1
        t_lst.append(t)

        # save prediction raw results
        if args.save_pred:
            prediction_path = os.path.join(results_raw_dir, f'{name}+{args.algo}@{i}.pkl')
            f = open(prediction_path, 'wb')
            pickle.dump(predictions, f)
            f.close()

        # save raw results of evaluation metrics
        txt = f'{name},'
        txt += ', '.join(['%.4f' % a for a in eval_metrics]) + \
               ', pa, ' + \
               ', '.join(['%.4f' % a for a in adj_eval_metrics])
        txt += f', model, {args.algo}, time, {t:.1f} s, runs, {i+1}/{args.runs}'
        logger_fh.info(txt)
        logger_fh_raw.info(txt)

    avg_entry = np.average(np.array(entries), axis=0)
    std_entry = np.std(np.array(entries), axis=0)
    avg_t = np.average(np.array(t_lst))

    f1_lst.append(avg_entry[2])
    aupr_lst.append(avg_entry[1])

    txt = f'{name}, ' + ", ".join(['%.4f' % a for a in np.hstack([avg_entry, std_entry])]) + f', {avg_t:.1f} s'
    logger_fh.info(txt)
    logger_fh_avg.info(txt)


if args.record_avg:
    logger_fh.info(f'\nf1, {np.average(f1_lst):.4f}, aupr, {np.average(aupr_lst):.4f}, '
                   f'time, {(time.time()-start_time):.1f}')
    logger_fh_avg.info(f'\nf1, {np.average(f1_lst):.4f}, aupr, {np.average(aupr_lst):.4f}, '
                       f'time, {(time.time()-start_time):.1f}')

results_raw_dir_new = results_raw_dir.replace('raw-record', f'[done] raw-record')
os.rename(results_raw_dir, results_raw_dir_new)



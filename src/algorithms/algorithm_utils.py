import os
import random
import numpy as np
import torch
import string
mask = ''.join(random.sample(string.ascii_letters, 8))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, intermediate_dir,
                 patience=7, verbose=False, delta=5e-5, model_name="",
                 trace_func=print, structrue='torch'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.structure = structrue

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        if structrue == 'torch':
            self.path = os.path.join(intermediate_dir, model_name, model_name + "." + mask + '_checkpoint.pt')
        elif structrue == 'keras':
            self.path = os.path.join(intermediate_dir, model_name, model_name + "." + mask + '_checkpoint.pt')

        self.trace_func = trace_func

        os.makedirs(os.path.split(self.path)[0], exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.structure == 'torch':
            torch.save(model.state_dict(), self.path)
        elif self.structure == 'keras':
            model.save(self.path)
        self.val_loss_min = val_loss

def get_sub_seqs(x_arr, seq_len, stride=1, start_discont=np.array([])):
    """
    :param start_discont: the start points of each sub-part in case the x_arr is just multiple parts joined together
    :param x_arr: dim 0 is time, dim 1 is channels
    :param seq_len: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - seq_len + 1), start)) for start in start_discont if start > seq_len]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - seq_len + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
    return x_seqs

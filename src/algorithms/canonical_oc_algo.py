import pandas as pd
import numpy as np
import torch
import random
import os
from numpy.random import RandomState
from torch.utils.data import DataLoader, Dataset
from .algorithm_utils import get_sub_seqs, EarlyStopping
from src.algorithms.net import NetModule


class Canonical:
    def __init__(self, sequence_length=100, stride=1,
                 num_epochs=40, batch_size=64, lr=1e-4,
                 hidden_dims=16, emb_dim=16, rep_hidden=16,
                 kernel_size=2, dropout=0.0, bias=True,
                 seed=0, es=False, device='cuda',
                 data_name=None, logger=None, model_dir='couta_model/',
                 # useless parameters, consistent with the proposed model
                 pretext_hidden=10, alpha=0.1, neg_batch_ratio=0.2, **others,
                 ):
        self.seq_len = sequence_length
        self.stride = stride

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device

        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.emb_dim = emb_dim
        self.rep_hidden = rep_hidden
        self.pretext_hidden = pretext_hidden
        self.dropout = dropout
        self.bias = bias

        self.seed = seed
        self.es = es
        self.train_val_pc = 0.25
        self.data_name = data_name
        self.log_func = logger.info if logger is not None else print
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.net = None
        self.c = None

        param_lst = locals()
        del param_lst['self']
        del param_lst['device']
        del param_lst['logger']
        del param_lst['data_name']
        del param_lst['alpha']
        del param_lst['neg_batch_ratio']
        del param_lst['pretext_hidden']

        self.log_func(param_lst)

        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        return

    def fit(self, X: pd.DataFrame):
        dim = X.shape[1]
        sequences = get_sub_seqs(X.values, seq_len=self.seq_len, stride=self.stride)
        sequences = sequences[RandomState(42).permutation(len(sequences))]

        if self.train_val_pc > 0:
            val_seqs = sequences[-int(self.train_val_pc * len(sequences)):]
            train_seqs = sequences[: -int(self.train_val_pc * len(sequences))]
        else:
            train_seqs = sequences
            val_seqs = None

        self.net = NetModule(
            input_dim=dim,
            hidden_dims=self.hidden_dims,
            emb_dim=self.emb_dim,
            pretext_hidden=self.pretext_hidden,
            rep_hidden=self.rep_hidden,
            out_dim=1,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            linear_bias=self.bias,
            tcn_bias=self.bias,
            pretext=False,
            dup=False
        )

        self.net.to(self.device)
        self.set_c(train_seqs)
        self.net = self.train(self.net, train_seqs, val_seqs)
        return

    def train(self, net, train_seqs, val_seqs=None):
        val_loader = DataLoader(dataset=SubseqData(val_seqs), batch_size=self.batch_size,
                                drop_last=False, shuffle=False)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion_dsvdd = DSVDDLoss(c=self.c)
        early_stp = EarlyStopping(intermediate_dir=self.model_dir,
                                  patience=7, model_name=self.__class__.__name__, verbose=False)

        net.train()
        for i in range(self.num_epochs):
            train_loader = DataLoader(dataset=SubseqData(train_seqs), batch_size=self.batch_size,
                                      drop_last=False, pin_memory=True, shuffle=True)
            train_loss = []
            for x in train_loader:
                x = x.float().to(self.device)
                rep = net(x)
                loss = criterion_dsvdd(rep)

                net.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.cpu().data.item()

                train_loss.append(loss)
            train_loss = np.mean(train_loss)

            # Get Validation loss
            val_loss = np.NAN
            if val_loader is not None:
                net.eval()
                val_loss = []
                with torch.no_grad():
                    for x in val_loader:
                        x = x.float().to(self.device)
                        rep = net(x)

                        loss = criterion_dsvdd(rep)
                        loss = loss.cpu().data.item()

                        val_loss.append(loss)
                val_loss = np.mean(val_loss)

            self.log_func(f'epoch: {i+1:02}, train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}')
            # pbar.set_postfix(epoch_loss=f'{np.mean(train_loss):.6f}')

            if self.es:
                early_metric = val_loss if val_loader is not None else train_loss
                early_stp(early_metric, model=net)

                if early_stp.early_stop:
                    net.load_state_dict(torch.load(early_stp.path))
                    self.log_func("early stop")
                    break
        return net

    def predict(self, X):
        data = X.values
        test_sub_seqs = get_sub_seqs(data, seq_len=self.seq_len, stride=1)

        test_dataset = SubseqData(test_sub_seqs)
        dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

        rep_lst = []
        self.net.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to('cuda')
                rep = self.net(x)
                rep_lst.append(rep)
        rep_emb = torch.cat(rep_lst)

        rep_score_ = torch.sum((rep_emb - self.c) ** 2, dim=1).data.cpu().numpy()
        rep_score_pad = np.hstack([0 * np.ones(data.shape[0] - rep_score_.shape[0]), rep_score_])

        predictions_dic = {'score_t': rep_score_pad,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': None,
                           'recons_tc': None,
                           }
        return predictions_dic

    def set_c(self, seqs, eps=0.1):
        """Initializing the center for the hypersphere"""
        dataloader = DataLoader(dataset=SubseqData(seqs), batch_size=self.batch_size,
                                  drop_last=True, pin_memory=True, shuffle=True)
        z_ = []
        self.net.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                rep = self.net(x)
                z_.append(rep.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.c = c



class SubseqData(Dataset):
    def __init__(self, x, y=None, w1=None, w2=None):
        self.sub_seqs = x
        self.label = y
        self.sample_weight1 = w1
        self.sample_weight2 = w2

    def __len__(self):
        return len(self.sub_seqs)

    def __getitem__(self, idx):
        if self.label is not None and self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.label[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        if self.label is not None:
            return self.sub_seqs[idx], self.label[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is None:
            return self.sub_seqs[idx], self.sample_weight[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        return self.sub_seqs[idx]


class DSVDDLoss(torch.nn.Module):
    def __init__(self, c, reduction='mean'):
        super(DSVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, sample_weight=None):
        loss = torch.sum((rep - self.c) ** 2, dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
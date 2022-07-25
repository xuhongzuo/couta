import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 bias=True, dropout=0.2, residual=True):
        """
        Residual block

        :param n_inputs: int, input channels
        :param n_outputs: int, output channels
        :param kernel_size: int, convolutional kernel size
        :param stride: int,
        :param dilation: int,
        :param padding: int,
        :param dropout: float, dropout
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, bias=bias,
                                           dilation=dilation))

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, bias=bias,
                                           dilation=dilation))
        self.Chomp1d = Chomp1d(padding)
        self.dropout = torch.nn.Dropout(dropout)

        self.residual = residual

        self.net = nn.Sequential(self.conv1, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
                                 self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()



    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        if self.residual:
            res = x if self.downsample is None else self.downsample(x)
            return out+res
        else:
            return out


class NetModule(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=32, rep_hidden=32, pretext_hidden=16,
                 emb_dim=10, kernel_size=2, dropout=0.2, out_dim=2,
                 tcn_bias=True, linear_bias=True,
                 dup=False, pretext=False):
        super(NetModule, self).__init__()

        self.layers = []

        if type(hidden_dims) == int: hidden_dims = [hidden_dims]
        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                          stride=1, dilation=dilation_size,
                                          padding=padding_size, dropout=dropout,
                                          bias=tcn_bias, residual=True)]
        self.network = nn.Sequential(*self.layers)
        self.l1 = nn.Linear(hidden_dims[-1], rep_hidden, bias=linear_bias)
        self.l2 = nn.Linear(rep_hidden, emb_dim, bias=linear_bias)
        self.act = torch.nn.LeakyReLU()

        self.dup = dup
        self.pretext = pretext

        if dup:
            self.l1_dup = nn.Linear(hidden_dims[-1], rep_hidden, bias=linear_bias)

        if pretext:
            self.pretext_l1 = nn.Linear(hidden_dims[-1], pretext_hidden, bias=linear_bias)
            self.pretext_l2 = nn.Linear(pretext_hidden, out_dim, bias=linear_bias)

    def forward(self, x):
        out = self.network(x.transpose(2, 1)).transpose(2, 1)
        out = out[:, -1]
        rep = self.l2(self.act(self.l1(out)))

        # pretext head
        if self.pretext:
            score = self.pretext_l2(self.act(self.pretext_l1(out)))

            if self.dup:
                rep_dup = self.l2(self.act(self.l1_dup(out)))
                return rep, rep_dup, score
            else:
                return rep, score

        else:
            if self.dup:
                rep_dup = self.l2(self.act(self.l1_dup(out)))
                return rep, rep_dup
            else:
                return rep

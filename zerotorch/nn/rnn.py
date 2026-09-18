"""循环网络：RNN（Elman）与 LSTM，均用引擎算子手写前向，自动获得反向。

输入约定：x 形状 (seq_len, batch, input_size)。
输出：outputs (seq_len, batch, hidden_size)，(h_n, c_n)。
"""
from __future__ import annotations

import math

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from .module import Module, Parameter
from . import init


def _init_rnn_weight(shape):
    t = Parameter(np.empty(shape))
    bound = 1.0 / math.sqrt(shape[-1])
    init.uniform_(t, -bound, bound)
    return t


class RNN(Module):
    """Elman RNN：h_t = tanh(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)。"""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        for l in range(num_layers):
            in_sz = input_size if l == 0 else hidden_size
            self.register_module(f'w_ih_{l}', _WeightPair(in_sz, hidden_size))

    def forward(self, x, state=None):
        seq, batch, _ = x.shape
        cur = x
        last_h = None
        for l in range(self.num_layers):
            wp = getattr(self, f'w_ih_{l}')
            h = np.zeros((batch, self.hidden_size), dtype=x.data.dtype)
            h = Tensor(h, requires_grad=True) if x.requires_grad else Tensor(h)
            layer_out = []
            for t in range(seq):
                xt = F.reshape(F.narrow(cur, 0, t, 1), (batch, -1))
                h = F.tanh(F.matmul(xt, F.transpose(wp.w)) + wp.b
                           + F.matmul(h, F.transpose(wp.w_hh)) + wp.b_hh)
                layer_out.append(h)
            cur = F.stack(layer_out, dim=0)
            last_h = h
        return cur, last_h


class LSTM(Module):
    """LSTM：i/f/g/o 四门，隐状态更新：
        c_t = f*c_{t-1} + i*g；h_t = o*tanh(c_t)
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        for l in range(num_layers):
            in_sz = input_size if l == 0 else hidden_size
            self.register_module(f'layer{l}', _LSTMLayer(in_sz, hidden_size, bias))

    def forward(self, x, state=None):
        seq, batch, _ = x.shape
        h = np.zeros((batch, self.hidden_size), dtype=x.data.dtype)
        c = np.zeros((batch, self.hidden_size), dtype=x.data.dtype)
        h = Tensor(h, requires_grad=True) if x.requires_grad else Tensor(h)
        c = Tensor(c, requires_grad=True) if x.requires_grad else Tensor(c)
        outputs = []
        for t in range(seq):
            xt = F.reshape(F.narrow(x, 0, t, 1), (batch, -1))
            h, c = self.layer0(xt, h, c)
            outputs.append(h)
        return F.stack(outputs, dim=0), (h, c)


class _WeightPair(Module):
    """RNN 单层权重：W_ih(hidden, input), b_ih, W_hh(hidden, hidden), b_hh。"""

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.w = _init_rnn_weight((hidden_size, input_size))
        self.w_hh = _init_rnn_weight((hidden_size, hidden_size))
        self.b = Parameter(np.zeros(hidden_size))
        self.b_hh = Parameter(np.zeros(hidden_size))


class _LSTMLayer(Module):
    """LSTM 单层：gate = [i; f; g; o]，一次矩阵乘后按 hidden 宽度切开。"""

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_ih = _init_rnn_weight((4 * hidden_size, input_size))
        self.w_hh = _init_rnn_weight((4 * hidden_size, hidden_size))
        self.b_ih = Parameter(np.zeros(4 * hidden_size))
        self.b_hh = Parameter(np.zeros(4 * hidden_size))

    def forward(self, x, h, c):
        hsz = self.hidden_size
        gates = (F.matmul(x, F.transpose(self.w_ih)) + self.b_ih
                 + F.matmul(h, F.transpose(self.w_hh)) + self.b_hh)
        i = F.narrow(gates, 1, 0 * hsz, hsz)
        f = F.narrow(gates, 1, 1 * hsz, hsz)
        g = F.narrow(gates, 1, 2 * hsz, hsz)
        o = F.narrow(gates, 1, 3 * hsz, hsz)
        c = F.sigmoid(f) * c + F.sigmoid(i) * F.tanh(g)
        h = F.sigmoid(o) * F.tanh(c)
        return h, c

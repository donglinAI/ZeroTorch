"""常用网络层：Linear / Conv2d / Embedding / Dropout / 归一化 / 池化 / 激活模块。"""
from __future__ import annotations

import math

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from .module import Module, Parameter
from . import init


class Linear(Module):
    """全连接层：y = x @ W.T + b。"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.empty((out_features, in_features)))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = Parameter(np.empty(out_features))
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        out = F.matmul(x, F.transpose(self.weight))
        if self.bias is not None:
            out = out + self.bias
        return out


class Conv2d(Module):
    """二维卷积：y = conv2d(x, W) + b。"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Parameter(np.empty(
            (out_channels, in_channels) + self.kernel_size))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = Parameter(np.empty(out_channels))
            fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class Embedding(Module):
    """嵌入层：查表 table[index]。"""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(np.empty((num_embeddings, embedding_dim)))
        init.normal_(self.weight, 0.0, 1.0)

    def forward(self, index):
        if not isinstance(index, np.ndarray):
            index = np.asarray(index)
        return F.embedding(self.weight, index)


class Dropout(Module):
    """Dropout：训练时按概率 p 置零并缩放 1/(1-p)。"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, self.training)


class LayerNorm(Module):
    """LayerNorm（对最后一维归一化）。"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) \
            else tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(self.normalized_shape))
        self.bias = Parameter(np.zeros(self.normalized_shape))

    def forward(self, x):
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class BatchNorm1d(Module):
    """BatchNorm，支持 (N, C) 与 (N, C, L)。"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))
        else:
            self.weight = None
            self.bias = None
        self.register_buffer('running_mean', np.zeros(num_features))
        self.register_buffer('running_var', np.ones(num_features))

    def forward(self, x):
        if self.weight is None:
            weight = np.ones(self.num_features, dtype=x.data.dtype)
            bias = np.zeros(self.num_features, dtype=x.data.dtype)
        else:
            weight, bias = self.weight, self.bias
        return F.batch_norm(
            x, weight, bias, self.eps, self.training,
            self.running_mean, self.running_var, self.momentum)


class BatchNorm2d(BatchNorm1d):
    """BatchNorm for (N, C, H, W)。内部 reshape 成 (N, C, H*W) 复用 BatchNorm1d 逻辑。"""

    def forward(self, x):
        batch, C, H, W = x.data.shape
        x = F.reshape(x, (batch, C, H * W))
        x = super().forward(x)
        return F.reshape(x, (batch, C, H, W))


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class Flatten(Module):
    """从 start_dim 起展平。"""

    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        return F.flatten(x, self.start_dim)


# ------------------------- 激活函数模块 -------------------------
class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return F.tanh(x)


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class SiLU(Module):
    def forward(self, x):
        return F.silu(x)


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, self.dim)

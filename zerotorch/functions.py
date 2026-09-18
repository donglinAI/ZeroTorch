"""函数式 API（对标 torch.nn.functional）：算子的一层薄封装。

用法::

    import zerotorch as nf
    y = nf.relu(nf.matmul(x, w.T) + b)
    loss = nf.cross_entropy(logits, targets)
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor
from . import ops


# ------------------------- 基础运算 -------------------------
def add(a, b):
    return ops.Add.apply(a, b)


def sub(a, b):
    return ops.Sub.apply(a, b)


def mul(a, b):
    return ops.Mul.apply(a, b)


def div(a, b):
    return ops.Div.apply(a, b)


def pow_(a, exponent):
    return ops.Pow.apply(a, exponent)


def neg(a):
    return ops.Neg.apply(a)


def exp(a):
    return ops.Exp.apply(a)


def log(a):
    return ops.Log.apply(a)


def sqrt(a):
    return ops.Sqrt.apply(a)


def abs_(a):
    return ops.Abs.apply(a)


def where(cond, a, b):
    return ops.Where.apply(cond, a, b)


def maximum(a, b):
    return ops.Maximum.apply(a, b)


def minimum(a, b):
    return ops.Minimum.apply(a, b)


# ------------------------- 矩阵 / 形状 -------------------------
def matmul(a, b):
    return ops.Matmul.apply(a, b)


def reshape(a, shape):
    return ops.Reshape.apply(a, shape)


def flatten(a, start_dim=1, end_dim=-1):
    return ops.Flatten.apply(a, start_dim, end_dim)


def transpose(a, axes=None):
    return ops.Transpose.apply(a, axes)


def permute(a, axes):
    return ops.Transpose.apply(a, axes)


def broadcast_to(a, shape):
    return ops.BroadcastTo.apply(a, shape)


def sum(a, dim=None, keepdims=False):
    return ops.Sum.apply(a, dim, keepdims)


def mean(a, dim=None, keepdims=False):
    return ops.Mean.apply(a, dim, keepdims)


def max(a, dim=None, keepdims=False):
    return ops.Max.apply(a, dim, keepdims)


def concat(arrays, dim=0):
    return ops.Concat.apply(list(arrays), dim)


def stack(arrays, dim=0):
    return ops.Stack.apply(list(arrays), dim)


def narrow(a, dim, start, length):
    return ops.Narrow.apply(a, dim, start, length)


def squeeze(a, dim=None):
    return ops.Squeeze.apply(a, dim)


def unsqueeze(a, dim):
    return ops.Unsqueeze.apply(a, dim)


def one_hot(index, num_classes):
    return ops.OneHot.apply(index, num_classes)


# ------------------------- 激活函数 -------------------------
def relu(a):
    return ops.Relu.apply(a)


def leaky_relu(a, alpha=0.01):
    return ops.LeakyRelu.apply(a, alpha)


def sigmoid(a):
    return ops.Sigmoid.apply(a)


def tanh(a):
    return ops.Tanh.apply(a)


def softmax(a, dim=-1):
    return ops.Softmax.apply(a, dim)


def log_softmax(a, dim=-1):
    return ops.LogSoftmax.apply(a, dim)


def gelu(a):
    return ops.Gelu.apply(a)


def silu(a):
    return ops.Silu.apply(a)


# ------------------------- 卷积 / 池化 / 归一化 -------------------------
def conv2d(x, weight, bias=None, stride=1, padding=0):
    return ops.Conv2d.apply(x, weight, bias, stride, padding)


def max_pool2d(x, kernel_size, stride=None, padding=0):
    return ops.MaxPool2d.apply(x, kernel_size, stride, padding)


def avg_pool2d(x, kernel_size, stride=None, padding=0):
    return ops.AvgPool2d.apply(x, kernel_size, stride, padding)


def upsample(x, scale):
    """最近邻上采样（整数倍 scale）。"""
    return ops.Upsample.apply(x, scale)


def layer_norm(x, gamma, beta, eps=1e-5):
    return ops.LayerNorm.apply(x, gamma, beta, eps)


def batch_norm(x, gamma, beta, eps=1e-5, training=True,
               running_mean=None, running_var=None, momentum=0.1):
    return ops.BatchNorm.apply(x, gamma, beta, eps, training,
                               running_mean, running_var, momentum)


def embedding(table, index):
    return ops.Take.apply(table, index)


def gaussian_sample(mu, logvar):
    return ops.GaussianSample.apply(mu, logvar)


# ------------------------- 损失函数 -------------------------
def cross_entropy(logits, targets, ignore_index=-100):
    return ops.CrossEntropy.apply(logits, targets, ignore_index)


def binary_cross_entropy_with_logits(logits, targets):
    return ops.BCEWithLogits.apply(logits, targets)


def binary_cross_entropy(prob, targets, eps=1e-7):
    return ops.BinaryCrossEntropy.apply(prob, targets, eps)


def smooth_l1_loss(pred, target, beta=1.0):
    return ops.SmoothL1.apply(pred, target, beta)


def mse_loss(pred, target):
    diff = pred - target
    return mean(diff * diff)


def l1_loss(pred, target):
    return mean(abs_(pred - target))


def kl_div_gaussian(mu, logvar):
    """KL(N(mu, exp(logvar)) || N(0, 1))，对 batch 求平均。"""
    kld = -0.5 * sum(1 + logvar - mu * mu - exp(logvar), dim=1)
    return mean(kld)


def dropout(x, p=0.5, training=True):
    """训练态：以概率 p 置零并乘以 1/(1-p) 保持期望；否则恒等。"""
    if not training or p <= 0:
        return x
    mask = (np.random.rand(*x.shape) > p) / (1.0 - p)
    return ops.Mul.apply(x, mask.astype(x.dtype))

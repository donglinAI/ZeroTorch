"""卷积与池化算子：Conv2d（im2col 实现）、MaxPool2d。

反向传播推导（Conv2d）：
- dW = dout 与 col 的矩阵乘；db = dout 求和；
- dX = dout 与 W 的矩阵乘得到 dcol，再经 col2im 散射回输入（重叠区累加）。
"""
from __future__ import annotations

import numpy as np

from ..tensor import Function


def _im2col(x, kh, kw, stride, pad):
    """把 (N, C, H, W) 的输入展开成 (N, C*kh*kw, H_out*W_out) 的列矩阵。"""
    N, C, H, W = x.shape
    H_out = (H + 2 * pad - kh) // stride + 1
    W_out = (W + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.zeros((N, C, kh, kw, H_out, W_out), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = xp[:, :, i:i + stride * H_out:stride,
                                        j:j + stride * W_out:stride]
    return cols.reshape(N, C * kh * kw, H_out * W_out), H_out, W_out


def _col2im(cols, x_shape, kh, kw, stride, pad):
    """把 (N, C*kh*kw, H_out*W_out) 的梯度散射回 (N, C, H, W)，重叠区累加。"""
    N, C, H, W = x_shape
    _, _, L = cols.shape
    H_out = (H + 2 * pad - kh) // stride + 1
    W_out = L // H_out
    cols = cols.reshape(N, C, kh, kw, H_out, W_out)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + stride * H_out:stride,
               j:j + stride * W_out:stride] += cols[:, :, i, j]
    if pad:
        return xp[:, :, pad:-pad, pad:-pad]
    return xp


class Conv2d(Function):
    """二维卷积。apply(x, weight, bias, stride, padding)。

    x: (N, C_in, H, W)；weight: (C_out, C_in, kh, kw)；bias: (C_out,) 或 None。
    """

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        N, C, H, W = x.shape
        C_out, C_in, kh, kw = weight.shape
        cols, H_out, W_out = _im2col(x, kh, kw, stride, padding)
        w_flat = weight.reshape(C_out, C_in * kh * kw)
        out = (w_flat @ cols).reshape(N, C_out, H_out, W_out)
        if bias is not None:
            out = out + bias.reshape(1, C_out, 1, 1)
        ctx.save_for_backward(
            x_shape=x.shape, weight=weight, cols=cols,
            stride=stride, padding=padding, kh=kh, kw=kw,
            N=N, C_out=C_out, H_out=H_out, W_out=W_out, has_bias=bias is not None)
        return out

    @staticmethod
    def backward(ctx, g):
        x_shape, weight = ctx['x_shape'], ctx['weight']
        cols = ctx['cols']
        N, C_out, H_out, W_out = ctx['N'], ctx['C_out'], ctx['H_out'], ctx['W_out']
        stride, padding = ctx['stride'], ctx['padding']
        kh, kw = ctx['kh'], ctx['kw']
        g_flat = g.reshape(N, C_out, H_out * W_out)

        dW = (g_flat @ cols.transpose(0, 2, 1)).sum(axis=0)          # (C_out, C_in*kh*kw)
        dW = dW.reshape(weight.shape)
        dcol = weight.reshape(C_out, -1).T @ g_flat                  # (C_in*kh*kw, L)
        dX = _col2im(dcol, x_shape, kh, kw, stride, padding)
        grads = [dX, dW]
        if ctx['has_bias']:
            grads.append(g.sum(axis=(0, 2, 3)))
        else:
            grads.append(None)
        grads += [None, None]
        return grads


class MaxPool2d(Function):
    """二维最大池化（padding=0）。apply(x, kernel_size, stride)。"""

    @staticmethod
    def forward(ctx, x, kernel_size, stride=None, padding=0):
        if padding != 0:
            raise NotImplementedError('MaxPool2d 暂不支持 padding，请先 pad 输入。')
        N, C, H, W = x.shape
        kh = kw = kernel_size
        s = stride if stride is not None else kernel_size
        H_out = (H - kh) // s + 1
        W_out = (W - kw) // s + 1
        windows = np.zeros((N, C, H_out, W_out, kh, kw), dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                windows[:, :, :, :, i, j] = x[:, :, i:i + s * H_out:s,
                                              j:j + s * W_out:s]
        values = windows.max(axis=(-2, -1))
        indices = windows.reshape(N, C, H_out, W_out, kh * kw).argmax(axis=-1)
        ctx.save_for_backward(
            x_shape=x.shape, kh=kh, kw=kw, stride=s,
            H_out=H_out, W_out=W_out, indices=indices)
        return values

    @staticmethod
    def backward(ctx, g):
        x_shape, kh, kw, stride = ctx['x_shape'], ctx['kh'], ctx['kw'], ctx['stride']
        H_out, W_out = ctx['H_out'], ctx['W_out']
        indices = ctx['indices']                    # (N, C, H_out, W_out)
        N, C, H, W = x_shape
        dx = np.zeros(x_shape, dtype=g.dtype)
        idx_i = indices // kw
        idx_j = indices % kw
        for i in range(kh):
            for j in range(kw):
                mask = (idx_i == i) & (idx_j == j)
                dx[:, :, i:i + stride * H_out:stride,
                   j:j + stride * W_out:stride] += g * mask
        return dx, None, None, None


class AvgPool2d(Function):
    """二维平均池化（padding=0）。"""

    @staticmethod
    def forward(ctx, x, kernel_size, stride=None, padding=0):
        if padding != 0:
            raise NotImplementedError('AvgPool2d 暂不支持 padding。')
        N, C, H, W = x.shape
        kh = kw = kernel_size
        s = stride if stride is not None else kernel_size
        H_out = (H - kh) // s + 1
        W_out = (W - kw) // s + 1
        windows = np.zeros((N, C, H_out, W_out, kh, kw), dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                windows[:, :, :, :, i, j] = x[:, :, i:i + s * H_out:s,
                                              j:j + s * W_out:s]
        ctx.save_for_backward(
            x_shape=x.shape, kh=kh, kw=kw, stride=s, H_out=H_out, W_out=W_out)
        return windows.mean(axis=(-2, -1))

    @staticmethod
    def backward(ctx, g):
        x_shape, kh, kw, stride = ctx['x_shape'], ctx['kh'], ctx['kw'], ctx['stride']
        H_out, W_out = ctx['H_out'], ctx['W_out']
        N, C, H, W = x_shape
        dx = np.zeros(x_shape, dtype=g.dtype)
        scale = 1.0 / (kh * kw)
        for i in range(kh):
            for j in range(kw):
                dx[:, :, i:i + stride * H_out:stride,
                   j:j + stride * W_out:stride] += g * scale
        return dx, None, None, None

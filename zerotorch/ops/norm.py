"""归一化算子：LayerNorm 与 BatchNorm（1d/2d 通用），均以融合 Function 实现并手推反向。"""
from __future__ import annotations

import numpy as np

from ..tensor import Function


class LayerNorm(Function):
    """对最后一维（或 normalized_shape 覆盖的若干维）做归一化。

    apply(x, gamma, beta, eps)
      x:      (..., D)  任意形状
      gamma:  (D,)，归一化维度的可学习缩放
      beta:   (D,)，归一化维度的可学习偏置
    反向推导（设 y = xhat * gamma + beta，xhat = (x - mu)/sigma，对最后一维）：
      dxhat = dout * gamma
      dx = (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)) / sigma
      dgamma = sum(dout * xhat, dim=norm_dims)
      dbeta  = sum(dout, dim=norm_dims)
    """

    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        dims = tuple(range(x.ndim - gamma.ndim, x.ndim))     # 被归一化的维度
        mu = x.mean(axis=dims, keepdims=True)
        var = x.var(axis=dims, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + eps)
        out = xhat * gamma + beta
        ctx.save_for_backward(xhat=xhat, gamma=gamma,
                              inv_std=1.0 / np.sqrt(var + eps), dims=dims)
        return out

    @staticmethod
    def backward(ctx, g):
        xhat, gamma, inv_std, dims = ctx['xhat'], ctx['gamma'], ctx['inv_std'], ctx['dims']
        dy = g * gamma
        mean = lambda t: t.mean(axis=dims, keepdims=True)
        dxhat = dy - mean(dy) - xhat * mean(dy * xhat)
        dx = dxhat * inv_std
        reduce_axes = tuple(a for a in range(g.ndim) if a not in dims)
        dgamma = (g * xhat).sum(axis=reduce_axes)
        dbeta = g.sum(axis=reduce_axes)
        return dx, dgamma, dbeta, None


class BatchNorm(Function):
    """BatchNorm，支持 (N, C) 与 (N, C, L) 输入。

    apply(x, gamma, beta, eps, training, running_mean, running_var, momentum)
      running_mean / running_var 为 numpy 数组（模块 buffer），训练时原地更新。
    反向推导（训练态，用批量统计量）：
      dxhat = dout * gamma
      dx = (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)) / sigma
      dgamma = sum(dout * xhat, axis=(N, ...))；dbeta = sum(dout, axis=(N, ...))
    """

    @staticmethod
    def forward(ctx, x, gamma, beta, eps, training, running_mean, running_var, momentum):
        axes = (0,) + tuple(range(2, x.ndim))     # 除 (N, C) 外的维度
        g_shape = (1, -1) + (1,) * (x.ndim - 2)   # (1, C, 1, ...) 广播形状
        gamma_b = gamma.reshape(g_shape)
        beta_b = beta.reshape(g_shape)
        if training:
            batch_mean = x.mean(axis=axes, keepdims=True)
            batch_var = x.var(axis=axes, keepdims=True)
            rm = running_mean
            rv = running_var
            rm[:] = momentum * rm + (1 - momentum) * batch_mean.reshape(gamma.shape)
            rv[:] = momentum * rv + (1 - momentum) * batch_var.reshape(gamma.shape)
            mu, var = batch_mean, batch_var
        else:
            shape = (1, -1) + (1,) * (x.ndim - 2)
            mu = running_mean.reshape(shape)
            var = running_var.reshape(shape)
        inv_std = 1.0 / np.sqrt(var + eps)
        xhat = (x - mu) * inv_std
        out = xhat * gamma_b + beta_b
        ctx.save_for_backward(xhat=xhat, gamma=gamma, inv_std=inv_std, axes=axes)
        return out

    @staticmethod
    def backward(ctx, g):
        xhat, gamma, inv_std, axes = ctx['xhat'], ctx['gamma'], ctx['inv_std'], ctx['axes']
        g_shape = (1, -1) + (1,) * (g.ndim - 2)
        dy = g * gamma.reshape(g_shape)
        mean = lambda t: t.mean(axis=axes, keepdims=True)
        dxhat = dy - mean(dy) - xhat * mean(dy * xhat)
        dx = dxhat * inv_std
        dgamma = (g * xhat).sum(axis=axes)
        dbeta = g.sum(axis=axes)
        return dx, dgamma, dbeta, None, None, None, None, None

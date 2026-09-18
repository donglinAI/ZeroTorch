"""融合损失算子：数值稳定的 CrossEntropy / BCEWithLogits / BCE / SmoothL1。

以 Function 形式实现，便于在 functional 与 loss 模块中复用，反向推导集中在一处。
"""
from __future__ import annotations

import numpy as np

from ..tensor import Function


def _sigmoid(x):
    """数值稳定的 sigmoid。"""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class CrossEntropy(Function):
    """logits(N,C) + targets(N,) -> 标量损失（mean reduction，支持 ignore_index）。

    反向：dL/dx = (softmax - onehot) * mask / denom
    """

    @staticmethod
    def forward(ctx, logits, targets, ignore_index=-100):
        targets = np.asarray(targets)          # 防御：list 输入时 != 会退化为标量比较
        N, C = logits.shape
        m = logits.max(axis=1, keepdims=True)
        logp = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
        mask = (targets != ignore_index)
        t = np.clip(targets, 0, C - 1)
        loss = -logp[np.arange(N), t] * mask
        denom = max(int(mask.sum()), 1)
        ctx.save_for_backward(logp=logp, targets=t, mask=mask, denom=denom)
        return np.asarray(loss.sum() / denom)

    @staticmethod
    def backward(ctx, g):
        logp, t, mask, denom = ctx['logp'], ctx['targets'], ctx['mask'], ctx['denom']
        N = logp.shape[0]
        s = np.exp(logp)                       # softmax
        e = np.zeros_like(logp)
        e[np.arange(N), t] = 1.0
        e = e * mask[:, None]
        dx = (s * mask[:, None] - e) * (g / denom)
        return dx, None, None


class BCEWithLogits(Function):
    """x 与 0/1 标签 t 的二元交叉熵（mean reduction，数值稳定）。

    loss = max(x,0) - x*t + log(1+exp(-|x|))；反向 dL/dx = (sigmoid(x) - t)/N
    """

    @staticmethod
    def forward(ctx, logits, targets):
        x, t = logits, np.asarray(targets)
        loss = np.maximum(x, 0) - x * t + np.log1p(np.exp(-np.abs(x)))
        ctx.save_for_backward(sigmoid=_sigmoid(x), n=x.size, targets=t)
        return np.asarray(loss.mean())

    @staticmethod
    def backward(ctx, g):
        return (ctx['sigmoid'] - ctx['targets']) * g / ctx['n'], None


class BinaryCrossEntropy(Function):
    """概率形式 BCE：-(t*log(p) + (1-t)*log(1-p))，p 应落在 (0,1)（内部 clip 到 eps）。"""

    @staticmethod
    def forward(ctx, prob, targets, eps=1e-7):
        p = np.clip(prob, eps, 1 - eps)
        targets = np.asarray(targets)
        loss = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
        ctx.save_for_backward(p=p, n=loss.size, targets=targets)
        return np.asarray(loss.mean())

    @staticmethod
    def backward(ctx, g):
        p = ctx['p']
        grad = (p - ctx['targets']) / (p * (1 - p))
        return grad * g / ctx['n'], None, None


class SmoothL1(Function):
    """Huber 损失：|d| < beta 时 0.5*d^2/beta，否则 |d| - 0.5*beta（mean reduction）。"""

    @staticmethod
    def forward(ctx, pred, target, beta=1.0):
        d = pred - target
        a = np.abs(d)
        loss = np.where(a < beta, 0.5 * d * d / beta, a - 0.5 * beta)
        ctx.save_for_backward(d=d, a=a, beta=beta)
        return np.asarray(loss.mean())

    @staticmethod
    def backward(ctx, g):
        d, a, beta = ctx['d'], ctx['a'], ctx['beta']
        grad = np.where(a < beta, d / beta, np.sign(d)) * g / d.size
        return grad, -grad, None

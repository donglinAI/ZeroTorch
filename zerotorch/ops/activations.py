"""激活函数算子：ReLU / LeakyReLU / Sigmoid / Tanh / Softmax / GELU / SiLU。"""
from __future__ import annotations

import numpy as np

from ..tensor import Function

_SQRT_2_PI = 0.7978845608028654      # sqrt(2/pi)
_GELU_B = 0.044715


class Relu(Function):
    """ReLU(x) = max(x, 0)。"""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a=a)
        return np.maximum(a, 0.0)

    @staticmethod
    def backward(ctx, g):
        return g * (ctx['a'] > 0),


class LeakyRelu(Function):
    """LeakyReLU：x > 0 时 x，否则 alpha * x。"""

    @staticmethod
    def forward(ctx, a, alpha):
        ctx.save_for_backward(a=a, alpha=alpha)
        return np.where(a > 0, a, alpha * a)

    @staticmethod
    def backward(ctx, g):
        a, alpha = ctx['a'], ctx['alpha']
        return g * np.where(a > 0, 1.0, alpha), None


class Sigmoid(Function):
    """Sigmoid（数值稳定实现，避免 exp 溢出）。"""

    @staticmethod
    def forward(ctx, a):
        out = np.where(a >= 0,
                       1.0 / (1.0 + np.exp(-a)),
                       np.exp(a) / (1.0 + np.exp(a)))
        ctx.save_for_backward(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        out = ctx['out']
        return g * out * (1.0 - out),


class Tanh(Function):
    """双曲正切。"""

    @staticmethod
    def forward(ctx, a):
        out = np.tanh(a)
        ctx.save_for_backward(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        out = ctx['out']
        return g * (1.0 - out * out),


class Softmax(Function):
    """Softmax（沿指定维度，数值稳定）。"""

    @staticmethod
    def forward(ctx, a, dim=-1):
        dim = dim if dim >= 0 else a.ndim + dim
        m = a.max(axis=dim, keepdims=True)
        e = np.exp(a - m)
        out = e / e.sum(axis=dim, keepdims=True)
        ctx.save_for_backward(out=out, dim=dim)
        return out

    @staticmethod
    def backward(ctx, g):
        out, dim = ctx['out'], ctx['dim']
        s = (g * out).sum(axis=dim, keepdims=True)
        return out * (g - s), None


class LogSoftmax(Function):
    """LogSoftmax（数值稳定）。"""

    @staticmethod
    def forward(ctx, a, dim=-1):
        dim = dim if dim >= 0 else a.ndim + dim
        m = a.max(axis=dim, keepdims=True)
        logp = a - m - np.log(np.exp(a - m).sum(axis=dim, keepdims=True))
        ctx.save_for_backward(out=logp, dim=dim)
        return logp

    @staticmethod
    def backward(ctx, g):
        logp, dim = ctx['out'], ctx['dim']
        s = np.exp(logp)                       # 即 softmax
        return g - s * g.sum(axis=dim, keepdims=True), None


class Gelu(Function):
    """GELU（tanh 近似）：0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))。"""

    @staticmethod
    def forward(ctx, a):
        u = _SQRT_2_PI * (a + _GELU_B * a ** 3)
        t = np.tanh(u)
        ctx.save_for_backward(a=a, t=t)
        return 0.5 * a * (1.0 + t)

    @staticmethod
    def backward(ctx, g):
        a, t = ctx['a'], ctx['t']
        dt = _SQRT_2_PI * (1.0 + 3.0 * _GELU_B * a * a) * (1.0 - t * t)
        return g * (0.5 * (1.0 + t) + 0.5 * a * dt),


class Silu(Function):
    """SiLU / Swish：x * sigmoid(x)。"""

    @staticmethod
    def forward(ctx, a):
        s = np.where(a >= 0,
                     1.0 / (1.0 + np.exp(-a)),
                     np.exp(a) / (1.0 + np.exp(a)))
        ctx.save_for_backward(a=a, s=s)
        return a * s

    @staticmethod
    def backward(ctx, g):
        a, s = ctx['a'], ctx['s']
        # dy/dx = s + x * s * (1 - s)
        return g * (s + a * s * (1.0 - s)),

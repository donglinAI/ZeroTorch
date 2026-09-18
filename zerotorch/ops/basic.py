"""基础逐元素算子：四则运算、幂、指数对数、绝对值、比较选择等。"""
from __future__ import annotations

import numpy as np

from ..tensor import Function, _sum_to_shape


class Add(Function):
    """加法（支持广播，标量自动转 0 维数组）。"""

    @staticmethod
    def forward(ctx, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        ctx.save_for_backward(a_shape=a.shape, b_shape=b.shape)
        return a + b

    @staticmethod
    def backward(ctx, g):
        return _sum_to_shape(g, ctx['a_shape']), _sum_to_shape(g, ctx['b_shape'])


class Sub(Function):
    """减法（支持广播，标量自动转 0 维数组）。"""

    @staticmethod
    def forward(ctx, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        ctx.save_for_backward(a_shape=a.shape, b_shape=b.shape)
        return a - b

    @staticmethod
    def backward(ctx, g):
        return _sum_to_shape(g, ctx['a_shape']), _sum_to_shape(-g, ctx['b_shape'])


class Mul(Function):
    """逐元素乘法（支持广播，标量自动转 0 维数组）。"""

    @staticmethod
    def forward(ctx, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        ctx.save_for_backward(a=a, b=b, a_shape=a.shape, b_shape=b.shape)
        return a * b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx['a'], ctx['b']
        ga = _sum_to_shape(g * b, ctx['a_shape'])
        gb = _sum_to_shape(g * a, ctx['b_shape'])
        return ga, gb


class Div(Function):
    """逐元素除法（支持广播，标量自动转 0 维数组）。"""

    @staticmethod
    def forward(ctx, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        ctx.save_for_backward(a=a, b=b, a_shape=a.shape, b_shape=b.shape)
        return a / b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx['a'], ctx['b']
        ga = _sum_to_shape(g / b, ctx['a_shape'])
        gb = _sum_to_shape(-g * a / (b * b), ctx['b_shape'])
        return ga, gb


class Pow(Function):
    """幂运算，指数为标量。"""

    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a=a, exponent=exponent)
        return np.power(a, exponent)

    @staticmethod
    def backward(ctx, g):
        a, n = ctx['a'], ctx['exponent']
        return g * n * np.power(a, n - 1), None


class Neg(Function):
    """取负。"""

    @staticmethod
    def forward(ctx, a):
        return -a

    @staticmethod
    def backward(ctx, g):
        return -g,


class Exp(Function):
    """指数函数。"""

    @staticmethod
    def forward(ctx, a):
        out = np.exp(a)
        ctx.save_for_backward(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        return g * ctx['out'],


class Log(Function):
    """自然对数。"""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a=a)
        return np.log(a)

    @staticmethod
    def backward(ctx, g):
        return g / ctx['a'],


class Sqrt(Function):
    """平方根。"""

    @staticmethod
    def forward(ctx, a):
        out = np.sqrt(a)
        ctx.save_for_backward(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        return g / (2.0 * ctx['out']),


class Abs(Function):
    """绝对值。"""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(sign=np.sign(a))
        return np.abs(a)

    @staticmethod
    def backward(ctx, g):
        return g * ctx['sign'],


class Maximum(Function):
    """逐元素最大值（ReLU 的底层实现）。"""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return np.maximum(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx['a'], ctx['b']
        return g * (a >= b), g * (b > a)


class Minimum(Function):
    """逐元素最小值。"""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return np.minimum(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx['a'], ctx['b']
        return g * (a <= b), g * (b < a)


class Where(Function):
    """np.where(cond, a, b)，cond 为 numpy 布尔数组（非 Tensor）。"""

    @staticmethod
    def forward(ctx, cond, a, b):
        ctx.save_for_backward(cond=cond)
        return np.where(cond, a, b)

    @staticmethod
    def backward(ctx, g):
        cond = ctx['cond']
        return None, g * cond, g * (~cond)

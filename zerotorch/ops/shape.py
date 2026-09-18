"""形状 / 规约 / 索引类算子：reshape、transpose、sum、mean、max、concat、narrow、gather 等。"""
from __future__ import annotations

import numpy as np

from ..tensor import Function, _sum_to_shape


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a_shape=a.shape)
        return a.reshape(shape)

    @staticmethod
    def backward(ctx, g):
        return g.reshape(ctx['a_shape']), None


class Flatten(Function):
    """展平 [start_dim, end_dim] 区间，默认展平到 1 维。"""

    @staticmethod
    def forward(ctx, a, start_dim=1, end_dim=-1):
        shape = list(a.shape)
        s = start_dim if start_dim >= 0 else a.ndim + start_dim
        e = end_dim if end_dim >= 0 else a.ndim + end_dim
        flat = 1
        for i in range(s, e + 1):
            flat *= shape[i]
        new_shape = shape[:s] + [flat] + shape[e + 1:]
        ctx.save_for_backward(a_shape=a.shape)
        return a.reshape(new_shape)

    @staticmethod
    def backward(ctx, g):
        return g.reshape(ctx['a_shape']), None


class Transpose(Function):
    """转置 / 维度置换。"""

    @staticmethod
    def forward(ctx, a, axes=None):
        if axes is None:
            ctx.save_for_backward(axes=None)
            return a.T
        ctx.save_for_backward(axes=tuple(axes))
        return a.transpose(axes)

    @staticmethod
    def backward(ctx, g):
        axes = ctx['axes']
        if axes is None:
            return g.T, None
        return g.transpose(np.argsort(axes)), None


class BroadcastTo(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a_shape=a.shape)
        return np.broadcast_to(a, shape).copy()

    @staticmethod
    def backward(ctx, g):
        return _sum_to_shape(g, ctx['a_shape']), None


class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):
        ctx.save_for_backward(a_shape=a.shape, dim=dim, keepdims=keepdims)
        return a.sum(axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        a_shape, dim, keepdims = ctx['a_shape'], ctx['dim'], ctx['keepdims']
        dims = _normalize_dims(dim, len(a_shape))
        if not keepdims:
            g = g.reshape(_keepdims_shape(a_shape, dims))
        return np.broadcast_to(g, a_shape).copy(), None, None


class Mean(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):
        ctx.save_for_backward(a_shape=a.shape, dim=dim, keepdims=keepdims)
        return a.mean(axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        a_shape, dim, keepdims = ctx['a_shape'], ctx['dim'], ctx['keepdims']
        dims = _normalize_dims(dim, len(a_shape))
        if dim is None:
            count = int(np.prod(a_shape)) if a_shape else 1
        else:
            count = 1
            for d in dims:
                count *= a_shape[d]
        if not keepdims:
            g = g.reshape(_keepdims_shape(a_shape, dims))
        return np.broadcast_to(g / count, a_shape).copy(), None, None


def _keepdims_shape(a_shape, dims):
    """把规约维恢复为 1 的 keepdims 形状。"""
    return tuple(1 if i in dims else s for i, s in enumerate(a_shape))


class Max(Function):
    """沿 dim 取最大值（dim=None 时为全局 max）。"""

    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):
        ctx.save_for_backward(a_shape=a.shape, dim=dim, keepdims=keepdims)
        values = a.max(axis=dim, keepdims=True)
        indices = a.argmax(axis=dim, keepdims=True)
        ctx.save_for_backward(indices=indices)
        if not keepdims:
            values = np.squeeze(values, axis=dim) if dim is not None else values.reshape(())
        return values

    @staticmethod
    def backward(ctx, g):
        a_shape, dim, keepdims = ctx['a_shape'], ctx['dim'], ctx['keepdims']
        indices = ctx['indices']
        dx = np.zeros(a_shape, dtype=g.dtype)
        if dim is None:
            dx.flat[indices] = g
            return dx, None, None
        g2 = g
        if not keepdims:
            g2 = np.expand_dims(g, axis=dim)
        np.put_along_axis(dx, indices, g2, axis=dim)
        return dx, None, None


class Concat(Function):
    """拼接。用法：Concat.apply([t1, t2, ...], dim)。"""

    @staticmethod
    def forward(ctx, arrays, dim):
        dim = dim if dim >= 0 else arrays[0].ndim + dim
        ctx.save_for_backward(shapes=[a.shape for a in arrays], dim=dim)
        return np.concatenate(arrays, axis=dim)

    @staticmethod
    def backward(ctx, g):
        shapes, dim = ctx['shapes'], ctx['dim']
        sizes = [s[dim] for s in shapes]
        parts = np.split(g, np.cumsum(sizes)[:-1], axis=dim)
        return [p.reshape(s) for p, s in zip(parts, shapes)], None


class Stack(Function):
    """堆叠。用法：Stack.apply([t1, t2, ...], dim)。"""

    @staticmethod
    def forward(ctx, arrays, dim=0):
        dim = dim if dim >= 0 else arrays[0].ndim + dim + 1
        ctx.save_for_backward(shapes=[a.shape for a in arrays], dim=dim)
        return np.stack(arrays, axis=dim)

    @staticmethod
    def backward(ctx, g):
        shapes, dim = ctx['shapes'], ctx['dim']
        parts = np.split(g, len(shapes), axis=dim)
        return [p.reshape(s) for p, s in zip(parts, shapes)], None


class Narrow(Function):
    """沿 dim 截取 [start, start+length)，反向时把梯度放回原位（等价于 slice 视图的梯度）。"""

    @staticmethod
    def forward(ctx, a, dim, start, length):
        dim = dim if dim >= 0 else a.ndim + dim
        ctx.save_for_backward(a_shape=a.shape, dim=dim, start=start, length=length)
        sl = [slice(None)] * a.ndim
        sl[dim] = slice(start, start + length)
        return a[tuple(sl)]

    @staticmethod
    def backward(ctx, g):
        a_shape, dim, start, length = ctx['a_shape'], ctx['dim'], ctx['start'], ctx['length']
        dx = np.zeros(a_shape, dtype=g.dtype)
        sl = [slice(None)] * len(a_shape)
        sl[dim] = slice(start, start + length)
        dx[tuple(sl)] = g
        return dx, None, None, None


class Take(Function):
    """通用索引：table[index]，反向用 np.add.at 把梯度散射回查表（Embedding 的底层实现）。"""

    @staticmethod
    def forward(ctx, table, index):
        ctx.save_for_backward(index=index, table_shape=table.shape)
        return table[index]

    @staticmethod
    def backward(ctx, g):
        index, table_shape = ctx['index'], ctx['table_shape']
        dtable = np.zeros(table_shape, dtype=g.dtype)
        np.add.at(dtable, index, g)
        return dtable, None


class OneHot(Function):
    @staticmethod
    def forward(ctx, index, num_classes):
        return np.eye(num_classes, dtype=np.float32)[index]

    @staticmethod
    def backward(ctx, g):
        return None, None


class GaussianSample(Function):
    """重参数化采样：z = mu + eps * exp(0.5 * logvar)，eps ~ N(0, 1)。"""

    @staticmethod
    def forward(ctx, mu, logvar):
        eps = np.random.randn(*mu.shape).astype(mu.dtype)
        ctx.save_for_backward(eps=eps, logvar=logvar)
        return mu + eps * np.exp(0.5 * logvar)

    @staticmethod
    def backward(ctx, g):
        eps, logvar = ctx['eps'], ctx['logvar']
        return g, 0.5 * g * eps * np.exp(0.5 * logvar)


class Squeeze(Function):
    @staticmethod
    def forward(ctx, a, dim=None):
        ctx.save_for_backward(a_shape=a.shape)
        return np.squeeze(a, axis=dim)

    @staticmethod
    def backward(ctx, g):
        return g.reshape(ctx['a_shape']), None


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, a, dim):
        ctx.save_for_backward(a_shape=a.shape)
        return np.expand_dims(a, axis=dim)

    @staticmethod
    def backward(ctx, g):
        return g.reshape(ctx['a_shape']), None


def _normalize_dims(dim, ndim):
    """把 dim（int / tuple / None）归一化为元组。"""
    if dim is None:
        return tuple(range(ndim))
    if isinstance(dim, int):
        dim = (dim,)
    return tuple(d if d >= 0 else ndim + d for d in dim)


def transpose(a, axes=None):
    """转置便捷函数（供 tensor.T 使用）。"""
    return Transpose.apply(a, axes)

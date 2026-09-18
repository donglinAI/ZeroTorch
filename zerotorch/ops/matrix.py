"""矩阵乘法算子（支持 2D / 批量 / 广播 / 1D 混合场景）。"""
from __future__ import annotations

import numpy as np

from ..tensor import Function, _sum_to_shape


class Matmul(Function):
    """np.matmul 语义。反向传播按场景计算：2D 用转置公式，其余用 einsum。"""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return np.matmul(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx['a'], ctx['b']

        if a.ndim == 2 and b.ndim == 2:          # 标准矩阵乘
            return g @ b.T, a.T @ g
        if a.ndim == 1 and b.ndim == 1:          # 向量点积 -> 标量
            return g * b, g * a

        if a.ndim == 1:                          # 向量 @ 矩阵/批量矩阵
            ga = np.einsum('...k,...mk->m', g, b)      # out: (...,K)
            gb = np.einsum('...k,m->...mk', g, a)
            return _sum_to_shape(ga, a.shape), _sum_to_shape(gb, b.shape)

        if b.ndim == 1:                          # 矩阵/批量矩阵 @ 向量
            ga = np.einsum('...m,k->...mk', g, b)
            gb = np.einsum('...m,...mk->k', g, a)
            return _sum_to_shape(ga, a.shape), _sum_to_shape(gb, b.shape)

        # 批量矩阵乘（支持广播批量维）
        ga = np.matmul(g, b.swapaxes(-1, -2))
        gb = np.matmul(a.swapaxes(-1, -2), g)
        return _sum_to_shape(ga, a.shape), _sum_to_shape(gb, b.shape)

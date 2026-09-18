"""最近邻上采样算子（Upsample · nearest）。

前向：把 (N, C, H, W) 的输入按整数 scale 放大，每个输入像素复制成 scale×scale 输出块：
    out[n, c, i*scale:(i+1)*scale, j*scale:(j+1)*scale] = x[n, c, i, j]
反向：输出梯度按最近邻映射 scatter 回输入，每个输入位置接收其对应输出块
（scale² 个格子）的梯度之和：
    dx[n, c, i, j] = Σ_{a,b} g[n, c, i*scale+a, j*scale+b]

服务对象：YOLOv3-lite 的简化 FPN 上采样、DDPM 的 U-Net 解码器。
只支持整数倍 scale（学习展示档位足够；分数倍需插值权重，不在本模块范围）。
"""
from __future__ import annotations

import numpy as np

from ..tensor import Function


class Upsample(Function):
    """最近邻上采样。apply(x, scale)。

    x: (N, C, H, W)；scale: int，输出 (N, C, H*scale, W*scale)。
    """

    @staticmethod
    def forward(ctx, x, scale):
        N, C, H, W = x.shape
        # (N, C, H, 1, W, 1) -> broadcast 到 (N, C, H, scale, W, scale) -> reshape
        out = np.broadcast_to(
            x[:, :, :, None, :, None], (N, C, H, scale, W, scale)
        ).reshape(N, C, H * scale, W * scale)
        ctx.save_for_backward(scale=scale, H=H, W=W)
        return out

    @staticmethod
    def backward(ctx, g):
        scale = ctx['scale']
        H, W = ctx['H'], ctx['W']
        N, C = g.shape[0], g.shape[1]
        # 恢复 scale 维，按块求和收缩回 (N, C, H, W)
        g = g.reshape(N, C, H, scale, W, scale)
        dx = g.sum(axis=(3, 5))
        return dx, None

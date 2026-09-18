"""数据变换：ToTensor / Normalize / FlattenImages。"""
from __future__ import annotations

import numpy as np

from ..tensor import Tensor


class ToTensor:
    def __call__(self, arr):
        return Tensor(np.asarray(arr))


class Normalize:
    """按 mean / std 归一化（作用于 numpy 数组或 Tensor 的 data）。"""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, x):
        if isinstance(x, Tensor):
            x.data = (x.data - self.mean) / self.std
            return x
        return (np.asarray(x) - self.mean) / self.std


class FlattenImages:
    """把 (..., C, H, W) 展平为 (..., C*H*W)。"""

    def __call__(self, x):
        arr = x.data if isinstance(x, Tensor) else np.asarray(x)
        out = arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr
        return Tensor(out) if isinstance(x, Tensor) else out

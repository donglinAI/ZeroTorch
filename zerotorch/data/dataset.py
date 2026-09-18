"""数据管线：Dataset 基类与 TensorDataset。"""
from __future__ import annotations

import numpy as np

from ..tensor import Tensor


class Dataset:
    """数据集基类（对标 torch.utils.data.Dataset）。"""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    """把若干数组/张量按第一维对齐组成数据集（合成数据常用）。"""

    def __init__(self, *tensors):
        self.tensors = [t.data if isinstance(t, Tensor) else np.asarray(t)
                        for t in tensors]
        if not self.tensors:
            raise ValueError('TensorDataset 至少需要一个张量')
        n = len(self.tensors[0])
        for t in self.tensors:
            if len(t) != n:
                raise ValueError('所有张量的第一维长度必须一致')

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

"""DataLoader：按 batch 取样本并 collate 成 Tensor。"""
from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from .dataset import Dataset
from .sampler import RandomSampler, SequentialSampler


def default_collate(batch):
    """把样本列表堆叠成 Tensor。

    - 每个样本是单数组 / 元组（多个数组）；
    - 数值（标量）转 0 维数组；整数默认 int64，浮点默认 float32。
    """
    if isinstance(batch[0], (tuple, list)):
        n = len(batch[0])
        return tuple(default_collate([b[i] for b in batch]) for i in range(n))
    arrs = []
    for b in batch:
        if isinstance(b, Tensor):
            arrs.append(b.data)
        elif isinstance(b, (int, float, np.number)):
            arrs.append(np.asarray(b))
        else:
            arrs.append(np.asarray(b))
    if arrs[0].ndim == 0:
        stacked = np.stack(arrs)
    else:
        stacked = np.stack(arrs)
    return Tensor(stacked)


class DataLoader:
    """数据加载器（对标 torch.utils.data.DataLoader）。

    说明：为保持"纯 Python + 可读性"，暂未实现多进程取数
    （num_workers 参数保留，置 0 表示主进程串行）。
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 drop_last=False, collate_fn=None):
        if not isinstance(dataset, Dataset):
            raise TypeError('dataset 必须是 Dataset 实例')
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler if sampler is not None else \
            (RandomSampler(dataset) if shuffle else SequentialSampler(dataset))
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

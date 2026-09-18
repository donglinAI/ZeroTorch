"""采样器：SequentialSampler / RandomSampler。"""
from __future__ import annotations

import numpy as np


class Sampler:
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            order = np.random.permutation(n)
        else:
            order = np.arange(n)
        return iter(order.tolist())

    def __len__(self):
        return len(self.data_source)

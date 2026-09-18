"""参数初始化工具（对标 torch.nn.init）。"""
from __future__ import annotations

import math

import numpy as np

from .module import Parameter


def _calculate_fan(shape):
    """根据形状计算 fan_in / fan_out。"""
    if len(shape) == 2:                      # Linear: (out, in)
        fan_in, fan_out = shape[1], shape[0]
    elif len(shape) == 4:                    # Conv2d: (out, in, kh, kw)
        receptive = shape[2] * shape[3]
        fan_in, fan_out = shape[1] * receptive, shape[0] * receptive
    else:
        raise ValueError(f'无法计算 fan：{shape}')
    return fan_in, fan_out


def zeros_(t):
    t.data.fill(0)
    return t


def ones_(t):
    t.data.fill(1)
    return t


def normal_(t, mean=0.0, std=1.0):
    t.data = np.random.normal(mean, std, size=t.shape).astype(t.data.dtype)
    return t


def uniform_(t, a=-1.0, b=1.0):
    t.data = np.random.uniform(a, b, size=t.shape).astype(t.data.dtype)
    return t


def xavier_uniform_(t, gain=1.0):
    fan_in, fan_out = _calculate_fan(t.shape)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(t, -bound, bound)


def xavier_normal_(t, gain=1.0):
    fan_in, fan_out = _calculate_fan(t.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(t, 0.0, std)


def kaiming_uniform_(t, a=math.sqrt(5)):
    fan_in, _ = _calculate_fan(t.shape)
    bound = math.sqrt(6.0 / ((1 + a * a) * fan_in))
    return uniform_(t, -bound, bound)


def kaiming_normal_(t, a=0.0):
    fan_in, _ = _calculate_fan(t.shape)
    std = math.sqrt(2.0 / ((1 + a * a) * fan_in))
    return normal_(t, 0.0, std)


def constant_(t, value):
    t.data.fill(value)
    return t

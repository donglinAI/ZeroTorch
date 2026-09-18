"""优化器基类与注册机制。"""
from __future__ import annotations

import numpy as np

from ..tensor import Tensor

OPTIM_REGISTRY = {}


def register_optimizer(name):
    """注册优化器，支持按名字即插即用。"""
    def deco(cls):
        cls.name = name
        OPTIM_REGISTRY[name] = cls
        return cls
    return deco


class Optimizer:
    """优化器基类（对标 torch.optim.Optimizer）。

    - ``param_groups``：参数组，每组可配置不同的 lr / 超参；
    - ``zero_grad()``：清空梯度；
    - ``step()``：按策略更新参数（子类实现）；
    - ``state_dict() / load_state_dict()``：支持断点续训。
    """

    def __init__(self, params, defaults):
        if isinstance(params, Optimizer):
            params = params.param_groups[0]['params']
        self.param_groups = [{'params': [p for p in params], **defaults}]
        self.state = {}            # id(param) -> 状态 dict
        self._step_count = 0

    def add_param_group(self, param_group):
        self.param_groups.append(param_group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

    def step(self):
        raise NotImplementedError

    # ------------------------- 序列化 -------------------------
    def state_dict(self):
        """返回可序列化状态：参数组配置 + 每个参数的状态（以组内序号索引）。"""
        sd = {'param_groups': [
            {k: v for k, v in g.items() if k != 'params'} for g in self.param_groups
        ]}
        state = {}
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group['params']):
                if id(p) in self.state:
                    state[f'{gi}/{pi}'] = self.state[id(p)]
        sd['state'] = state
        sd['step_count'] = self._step_count
        return sd

    def load_state_dict(self, state_dict):
        groups_cfg = state_dict['param_groups']
        for g, cfg in zip(self.param_groups, groups_cfg):
            for k, v in cfg.items():
                g[k] = v
        self.state = {}
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group['params']):
                key = f'{gi}/{pi}'
                if key in state_dict['state']:
                    self.state[id(p)] = dict(state_dict['state'][key])
        self._step_count = state_dict.get('step_count', 0)


def clip_grad_norm_(params, max_norm, norm_type=2.0):
    """梯度全局裁剪（对标 torch.nn.utils.clip_grad_norm_），返回裁剪前总范数。"""
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    total = sum(np.sum(np.square(p.grad)) for p in params) ** 0.5
    clip_coef = max_norm / (total + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad = p.grad * clip_coef
    return float(total)

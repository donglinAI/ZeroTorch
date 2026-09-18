"""SGD / Adam / AdamW 优化器。"""
from __future__ import annotations

import numpy as np

from .base import Optimizer, register_optimizer


@register_optimizer('sgd')
class SGD(Optimizer):
    """随机梯度下降，支持 momentum / weight_decay / nesterov。"""

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        if nesterov and momentum <= 0:
            raise ValueError('nesterov 需要 momentum > 0')
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, nesterov=nesterov))

    def step(self):
        for group in self.param_groups:
            lr, momentum, wd, nesterov = (
                group['lr'], group['momentum'], group['weight_decay'], group['nesterov'])
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p.data
                if momentum != 0:
                    buf = self.state.setdefault(id(p), {}).get('buf')
                    if buf is None:
                        buf = np.zeros_like(g)
                    buf = momentum * buf + g
                    self.state[id(p)]['buf'] = buf
                    g = g + momentum * buf if nesterov else buf
                p.data = p.data - lr * g


@register_optimizer('adam')
class Adam(Optimizer):
    """Adam：自适应矩估计（L2 正则作用于梯度）。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        self._step_count += 1
        for group in self.param_groups:
            lr, betas, eps, wd = (group['lr'], group['betas'],
                                  group['eps'], group['weight_decay'])
            beta1, beta2 = betas
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p.data
                st = self.state.setdefault(id(p), {})
                m = st.get('m', np.zeros_like(g))
                v = st.get('v', np.zeros_like(g))
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)
                st['m'], st['v'] = m, v
                m_hat = m / (1 - beta1 ** self._step_count)
                v_hat = v / (1 - beta2 ** self._step_count)
                p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + eps)


@register_optimizer('adamw')
class AdamW(Optimizer):
    """AdamW：解耦权重衰减（decoupled weight decay）。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        self._step_count += 1
        for group in self.param_groups:
            lr, betas, eps, wd = (group['lr'], group['betas'],
                                  group['eps'], group['weight_decay'])
            beta1, beta2 = betas
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(id(p), {})
                m = st.get('m', np.zeros_like(g))
                v = st.get('v', np.zeros_like(g))
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)
                st['m'], st['v'] = m, v
                m_hat = m / (1 - beta1 ** self._step_count)
                v_hat = v / (1 - beta2 ** self._step_count)
                # 解耦：权重衰减与梯度无关，直接作用于参数
                if wd != 0:
                    p.data = p.data - lr * wd * p.data
                p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + eps)

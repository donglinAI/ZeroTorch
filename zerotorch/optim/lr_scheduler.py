"""学习率调度器：StepLR / CosineAnnealingLR / LambdaLR。"""
from __future__ import annotations

import math


class LRScheduler:
    """基类：每个 epoch 结束时调用 step() 更新各参数组的学习率。"""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        new_lrs = self.get_lr()
        for g, lr in zip(self.optimizer.param_groups, new_lrs):
            g['lr'] = lr


class StepLR(LRScheduler):
    """每隔 step_size 个 epoch 乘以 gamma。"""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.gamma ** (self.last_epoch // self.step_size)
                for base in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """余弦退火到 eta_min。"""

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return list(self.base_lrs)
        return [self.eta_min + (base - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base in self.base_lrs]


class LambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.lr_lambda(self.last_epoch) for base in self.base_lrs]

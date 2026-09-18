"""损失函数库（注册机制，新增损失即插即用）。"""
from __future__ import annotations

from .. import functions as F
from ..nn.module import Module

LOSS_REGISTRY = {}


def register_loss(name):
    """注册损失函数，Trainer 等可按名字实例化（多进程并行时依赖此机制）。"""
    def deco(cls):
        cls.name = name
        LOSS_REGISTRY[name] = cls
        return cls
    return deco


class Loss(Module):
    """损失函数基类。子类实现 forward(pred, target) -> 标量 Tensor。"""

    def forward(self, pred, target):
        raise NotImplementedError


@register_loss('mse')
class MSELoss(Loss):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


@register_loss('l1')
class L1Loss(Loss):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


@register_loss('smooth_l1')
class SmoothL1Loss(Loss):
    """Huber 损失，常用于检测框回归。"""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, self.beta)


@register_loss('cross_entropy')
class CrossEntropyLoss(Loss):
    """多分类交叉熵（logits + 整数标签，含 log-softmax 数值稳定实现）。"""

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return F.cross_entropy(pred, target, self.ignore_index)


@register_loss('bce_with_logits')
class BCEWithLogitsLoss(Loss):
    """二元交叉熵（logits + 0/1 标签，数值稳定）。"""

    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)


@register_loss('bce')
class BCELoss(Loss):
    """概率形式二元交叉熵。"""

    def forward(self, pred, target):
        return F.binary_cross_entropy(pred, target)


@register_loss('kl_gaussian')
class KLGaussianLoss(Loss):
    """VAE 用：KL(N(mu, diag(exp(logvar))) || N(0, I))。"""

    def forward(self, mu, logvar):
        return F.kl_div_gaussian(mu, logvar)

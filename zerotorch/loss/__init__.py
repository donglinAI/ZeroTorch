"""损失包。"""
from .losses import (Loss, register_loss, LOSS_REGISTRY,
                     MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss,
                     BCEWithLogitsLoss, BCELoss, KLGaussianLoss)

__all__ = [
    'Loss', 'register_loss', 'LOSS_REGISTRY',
    'MSELoss', 'L1Loss', 'SmoothL1Loss', 'CrossEntropyLoss',
    'BCEWithLogitsLoss', 'BCELoss', 'KLGaussianLoss',
]

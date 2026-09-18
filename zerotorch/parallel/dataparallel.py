"""单机数据并行：DataParallel（单 CPU 教学版）。

设计理念（对标 torch.nn.DataParallel 的接口与架构）：
- 保留 DataParallel 的完整接口（包装模型、参数委托、train_step 训练路径）；
- Trainer 检测到 is_data_parallel=True 时，会走 train_step 路径而非默认前反向；
- **当前为单 CPU 教学版**：train_step 直接在主进程本地执行前向+反向，
  不启动多进程、不做真正的梯度 all-reduce——目的是展示并行训练的接口与
  架构骨架，而非追求加速。

未来扩展点（接入 GPU / 多进程时只需替换 train_step 内部实现）：
- 把 batch 沿第 0 维切分到多个 worker（进程 / GPU）；
- 各 worker 用广播的最新参数本地算梯度；
- 主进程收集梯度做 all-reduce（逐参数平均），统一执行优化器更新。
"""
from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from ..nn.module import Module


class DataParallel(Module):
    """数据并行包装器（单 CPU 教学版）。用法::

        model = DataParallel(MLP(...), num_workers=2)
        trainer = Trainer(model, loss_fn, optimizer, loader, epochs=5)
        trainer.fit()

    说明：num_workers 仅作接口预留，当前版本不启动多进程，所有计算在主进程本地执行。
    """

    is_data_parallel = True

    def __init__(self, module, num_workers=1):
        super().__init__()
        if not isinstance(module, Module):
            raise TypeError('module 必须是 nn.Module 实例')
        self.module = module
        self.num_workers = max(1, int(num_workers))

    # ------------------------- 参数委托（与原模型行为一致） -------------------------
    def forward(self, x):
        """推理：本地执行。"""
        return self.module(x)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self, prefix=''):
        return self.module.named_parameters(prefix)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict)

    def train(self, mode=True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def zero_grad(self):
        self.module.zero_grad()
        return self

    def num_parameters(self, trainable_only=False):
        return self.module.num_parameters(trainable_only)

    def __repr__(self):
        return (f'DataParallel(num_workers={self.num_workers} [单CPU教学版],\n'
                f'  module={self.module.__class__.__name__})')

    # ------------------------- 并行训练路径（单 CPU 本地执行） -------------------------
    def train_step(self, x, y, loss_fn):
        """DataParallel 训练单步。

        教学版：直接在主进程本地执行前向+反向，等价于单卡行为。
        未来接入多进程/GPU 时，在此处替换为：分片 → 并行算梯度 → all-reduce → 回填。
        """
        pred = self.module(x)
        loss = loss_fn(pred, y)
        loss.backward()
        return loss

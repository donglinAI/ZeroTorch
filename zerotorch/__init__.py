"""ZeroTorch：纯 Python + NumPy 实现的 PyTorch 风格深度学习框架（教学版）。

包入口（L3 完整版）：自动微分核心 + nn 层 + 损失/优化器/数据/训练器/序列化/可视化/并行。

示例::

    import zerotorch as zt
    model = zt.nn.Sequential(zt.nn.Linear(8, 32), zt.nn.ReLU(), zt.nn.Linear(32, 3))
    loss_fn = zt.loss.CrossEntropyLoss()
    optimizer = zt.optim.Adam(model.parameters(), lr=1e-3)
    trainer = zt.Trainer(model, loss_fn, optimizer, train_loader, epochs=10,
                         metrics=['accuracy'], callbacks=[zt.History(), zt.ProgressBar()])
    history = trainer.fit()
    zt.viz.plot_curves(history, 'curves.png')
"""
from .tensor import Tensor, tensor
from . import functions as F
from . import ops
from . import nn
from . import loss
from . import optim
from . import data
from . import metrics
from . import serialization
from . import viz
from . import parallel
from .inference import InferenceEngine
from .quantization import quantize_model, dequantize_model, compute_quantization_error

# 训练器常用类直接提到包顶层，对标 PyTorch Lightning 的使用习惯
from .trainer import Trainer, Callback, History, ModelCheckpoint, ProgressBar
from .parallel import DataParallel

__version__ = '0.1.0'

__all__ = [
    # 核心
    'Tensor', 'tensor', 'F', 'ops',
    # 神经网络
    'nn',
    # 训练三件套
    'loss', 'optim', 'metrics',
    # 数据
    'data',
    # 训练与部署
    'serialization', 'viz', 'parallel',
    # 推理与量化
    'InferenceEngine', 'quantize_model', 'dequantize_model', 'compute_quantization_error',
    # 训练器顶层快捷
    'Trainer', 'Callback', 'History', 'ModelCheckpoint', 'ProgressBar',
    'DataParallel',
    # 版本
    '__version__',
]

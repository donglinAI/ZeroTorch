"""推理部署模块：InferenceEngine。

封装模型加载、batch 推理、预热、性能基准，提供类似 PyTorch model.eval() 的推理接口。
教学版：纯 CPU，不做算子融合/图优化，但提供完整的推理流程。
"""
from __future__ import annotations

import time
import numpy as np

from .tensor import Tensor
from .serialization import load_checkpoint
from .nn.module import Module


class InferenceEngine:
    """推理引擎：封装模型推理，提供性能统计和 batch 推理。

    Args:
        model: 已训练的模型（Module 子类）
        device: 设备（教学版只支持 'cpu'）
    """

    def __init__(self, model: Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()  # 推理模式（关闭 dropout 等）

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model: Module = None, device: str = 'cpu'):
        """从 checkpoint 加载模型。

        Args:
            checkpoint_path: checkpoint 文件路径
            model: 模型实例（如果为 None，尝试从 checkpoint 恢复）
            device: 设备

        Returns:
            InferenceEngine 实例
        """
        if model is None:
            raise ValueError('教学版需要传入 model 实例（暂不支持从 checkpoint 自动恢复模型结构）')
        load_checkpoint(checkpoint_path, model=model)
        return cls(model, device=device)

    def predict(self, x: Tensor) -> Tensor:
        """单样本/批次推理。

        Args:
            x: 输入 Tensor（可以是 batch）

        Returns:
            模型输出
        """
        with self._no_grad():
            return self.model(x)

    def predict_batch(self, x: Tensor, batch_size: int = 1) -> np.ndarray:
        """分 batch 推理（适合大输入，避免内存不足）。

        Args:
            x: 输入 Tensor（总样本数 N）
            batch_size: 每 batch 样本数

        Returns:
            所有样本的输出 numpy 数组
        """
        self.model.eval()
        n = x.data.shape[0]
        outputs = []
        for i in range(0, n, batch_size):
            batch_x = Tensor(x.data[i:i+batch_size])
            with self._no_grad():
                out = self.model(batch_x)
            outputs.append(out.data)
        return np.concatenate(outputs, axis=0)

    def benchmark(self, x: Tensor, num_warmup: int = 3, num_iters: int = 10) -> dict:
        """性能基准：统计推理延迟和吞吐量。

        Args:
            x: 输入 Tensor（单 batch）
            num_warmup: 预热次数
            num_iters: 统计次数

        Returns:
            {'latency_ms': 平均延迟, 'throughput': 每秒推理数, 'batch_size': batch 大小}
        """
        self.model.eval()
        batch_size = x.data.shape[0]

        # 预热
        for _ in range(num_warmup):
            with self._no_grad():
                _ = self.model(x)

        # 统计
        latencies = []
        for _ in range(num_iters):
            start = time.time()
            with self._no_grad():
                _ = self.model(x)
            latencies.append((time.time() - start) * 1000)  # ms

        avg_latency = np.mean(latencies)
        throughput = batch_size / (avg_latency / 1000)  # samples/sec

        return {
            'latency_ms': round(avg_latency, 2),
            'throughput': round(throughput, 1),
            'batch_size': batch_size,
            'num_iters': num_iters,
        }

    def model_size_mb(self) -> float:
        """模型大小（MB，FP32）。"""
        total_params = sum(p.data.size for p in self.model.parameters())
        return round(total_params * 4 / 1024 / 1024, 3)  # FP32 = 4 bytes

    def _no_grad(self):
        """上下文管理器：禁用梯度计算（推理时省内存省时间）。"""
        class NoGradContext:
            def __enter__(self_inner):
                self_inner._prev_requires = []
                for p in self.model.parameters():
                    self_inner._prev_requires.append(p.requires_grad)
                    p.requires_grad = False
                return self_inner

            def __exit__(self_inner, *args):
                for p, prev in zip(self.model.parameters(), self_inner._prev_requires):
                    p.requires_grad = prev
                return False
        return NoGradContext()

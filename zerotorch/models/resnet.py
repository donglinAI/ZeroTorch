"""ResNet 轻量版：图像分类。

架构（对标 ResNet-18，通道缩减）：stem(3×3 conv + BN + ReLU) → 4 个 stage（每个 stage 多个 BasicBlock）
→ 全局平均池化 → Flatten → Linear 分类头。

轻量配置（默认）：32×32 图像、通道 [16,32,64,128]、每 stage 2 个 block，参数量 ~800K。
超轻量配置：通道 [8,16,32,64]、每 stage 1 个 block，参数量 ~100K，CPU 快速验证。
"""
from __future__ import annotations

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from ..nn.module import Module
from ..nn.container import Sequential
from ..nn.layers import Conv2d, BatchNorm2d, ReLU, AvgPool2d, Flatten, Linear
from .registry import register_model


class BasicBlock(Module):
    """ResNet BasicBlock：两个 3×3 卷积 + 残差连接。

    x → conv1 → bn1 → relu → conv2 → bn2 → +(shortcut) → relu
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu = ReLU()

        # 捷径（shortcut）：如果尺寸或通道变化，用 1×1 卷积投影
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = out + identity
        return self.relu(out)


@register_model('resnet_lite')
class ResNetLite(Module):
    """ResNet 轻量版（图像分类）。

    Args:
        in_channels: 输入图像通道数（默认 3）
        num_classes: 分类类别数（默认 10）
        channels: 4 个 stage 的输出通道数（默认 [16, 32, 64, 128]，原版 ResNet-18 的 1/4）
        num_blocks: 每个 stage 的 BasicBlock 数量（默认 [2, 2, 2, 2]，对标 ResNet-18）
        img_size: 输入图像边长（默认 32，用于计算全局池化 kernel）
    """

    def __init__(self, in_channels=3, num_classes=10,
                 channels=(16, 32, 64, 128), num_blocks=(2, 2, 2, 2),
                 img_size=32):
        super().__init__()
        assert len(channels) == 4, 'channels 必须有 4 个元素（对应 4 个 stage）'
        assert len(num_blocks) == 4, 'num_blocks 必须有 4 个元素'

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = list(channels)
        self.num_blocks = list(num_blocks)

        # stem：3×3 卷积 + BN + ReLU（不用 7×7 大卷积和 maxpool，因为 32×32 小图）
        self.stem = Sequential(
            Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            BatchNorm2d(channels[0]),
            ReLU(),
        )

        # 4 个 stage
        self.stage1 = self._make_stage(channels[0], channels[0], num_blocks[0], stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], num_blocks[1], stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], num_blocks[2], stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], num_blocks[3], stride=2)

        # 全局平均池化：32 → 16 → 8 → 4，最后特征图 4×4
        final_size = img_size // 8  # 3 次 stride=2 下采样
        self.avgpool = AvgPool2d(kernel_size=final_size)
        self.flatten = Flatten()
        self.fc = Linear(channels[3], num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """构建一个 stage：第一个 block 可能下采样，后续 block 保持尺寸。"""
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)    # (batch, C, 1, 1)
        x = self.flatten(x)     # (batch, C)
        return self.fc(x)       # (batch, num_classes)

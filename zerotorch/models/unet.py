"""U-Net 轻量版：图像分割。

经典 U 形架构：编码器（下采样提取特征）+ 瓶颈 + 解码器（上采样恢复分辨率）+ skip connection。
轻量配置：32×32 灰度图、通道 [8,16,32,64]、3 次下采样，参数量 ~120K，CPU 可训练。
"""
from __future__ import annotations

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from ..nn.module import Module
from ..nn.layers import Conv2d, BatchNorm2d, ReLU, MaxPool2d
from .registry import register_model


class ConvBlock(Module):
    """U-Net 卷积块：Conv3×3 + BN + ReLU × 2。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


@register_model('unet_lite')
class UNetLite(Module):
    """U-Net 轻量版（图像分割）。

    Args:
        in_channels: 输入通道数（默认 1，灰度图）
        num_classes: 分割类别数（默认 3，圆形/方形/三角形）
        base_channels: 基础通道数（默认 8，逐层翻倍：8→16→32→64）
    """

    def __init__(self, in_channels=1, num_classes=3, base_channels=8):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # 编码器（下采样路径）
        self.enc1 = ConvBlock(in_channels, c1)     # 32×32
        self.pool1 = MaxPool2d(kernel_size=2)       # → 16×16
        self.enc2 = ConvBlock(c1, c2)               # 16×16
        self.pool2 = MaxPool2d(kernel_size=2)       # → 8×8
        self.enc3 = ConvBlock(c2, c3)               # 8×8
        self.pool3 = MaxPool2d(kernel_size=2)       # → 4×4

        # 瓶颈
        self.bottleneck = ConvBlock(c3, c4)          # 4×4

        # 解码器（上采样路径）+ skip connection
        self.up3 = lambda x: F.upsample(x, scale=2)   # 4→8
        self.dec3 = ConvBlock(c4 + c3, c3)             # concat enc3 (c3) + up (c4) → c3
        self.up2 = lambda x: F.upsample(x, scale=2)   # 8→16
        self.dec2 = ConvBlock(c3 + c2, c2)            # concat enc2 (c2) + up (c3) → c2
        self.up1 = lambda x: F.upsample(x, scale=2)    # 16→32
        self.dec1 = ConvBlock(c2 + c1, c1)            # concat enc1 (c1) + up (c2) → c1

        # 输出头：1×1 卷积，逐像素分类
        self.head = Conv2d(c1, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 编码器（保存 skip connection）
        e1 = self.enc1(x)          # (batch, c1, 32, 32)
        e2 = self.enc2(self.pool1(e1))  # (batch, c2, 16, 16)
        e3 = self.enc3(self.pool2(e2))  # (batch, c3, 8, 8)

        # 瓶颈
        b = self.bottleneck(self.pool3(e3))  # (batch, c4, 4, 4)

        # 解码器（上采样 + concat skip + 卷积）
        d3 = self.up3(b)                         # (batch, c4, 8, 8)
        d3 = F.concat([d3, e3], dim=1)           # (batch, c4+c3, 8, 8)
        d3 = self.dec3(d3)                        # (batch, c3, 8, 8)

        d2 = self.up2(d3)                         # (batch, c3, 16, 16)
        d2 = F.concat([d2, e2], dim=1)           # (batch, c3+c2, 16, 16)
        d2 = self.dec2(d2)                        # (batch, c2, 16, 16)

        d1 = self.up1(d2)                         # (batch, c2, 32, 32)
        d1 = F.concat([d1, e1], dim=1)           # (batch, c2+c1, 32, 32)
        d1 = self.dec1(d1)                        # (batch, c1, 32, 32)

        # 输出头
        return self.head(d1)                       # (batch, num_classes, 32, 32)

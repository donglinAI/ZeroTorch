"""DCGAN 轻量版（Deep Convolutional GAN）：图像生成。

经典 GAN 架构：Generator（生成器）从噪声生成图像，Discriminator（判别器）区分真假。
用 Upsample + Conv 模拟转置卷积（不需要 ConvTranspose2d），纯 CPU 可训练。

轻量配置：32×32 灰度图、z_dim=64、生成/判别各 3 层卷积，参数量 ~200K。
"""
from __future__ import annotations

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from ..nn.module import Module
from ..nn.layers import Linear, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Flatten
from .registry import register_model


class Generator(Module):
    """生成器：从随机噪声生成 32×32 灰度图。

    输入: (batch, z_dim) 噪声向量
    输出: (batch, 1, 32, 32) 生成图像
    """

    def __init__(self, z_dim=64, channels=[128, 64, 32]):
        super().__init__()
        self.z_dim = z_dim
        # 全连接：把噪声 reshape 成小特征图
        self.fc = Linear(z_dim, channels[0] * 4 * 4)
        self.bn_fc = BatchNorm2d(channels[0])
        self.relu_fc = ReLU()

        # 上采样块：Upsample(×2) + Conv
        self.up1 = lambda x: F.upsample(x, scale=2)  # 4→8
        self.conv1 = Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(channels[1])
        self.relu1 = ReLU()

        self.up2 = lambda x: F.upsample(x, scale=2)  # 8→16
        self.conv2 = Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(channels[2])
        self.relu2 = ReLU()

        self.up3 = lambda x: F.upsample(x, scale=2)  # 16→32
        self.conv3 = Conv2d(channels[2], 1, kernel_size=3, stride=1, padding=1)
        self.tanh = Sigmoid()  # 输出 [0, 1]

    def forward(self, z):
        batch = z.data.shape[0]
        x = self.fc(z)
        x = F.reshape(x, (batch, -1, 4, 4))
        x = self.relu_fc(self.bn_fc(x))

        x = self.relu1(self.bn1(self.conv1(self.up1(x))))
        x = self.relu2(self.bn2(self.conv2(self.up2(x))))
        x = self.tanh(self.conv3(self.up3(x)))  # (batch, 1, 32, 32)
        return x


class Discriminator(Module):
    """判别器：判断图像是真还是假。

    输入: (batch, 1, 32, 32) 图像
    输出: (batch,) 判别概率（sigmoid）
    """

    def __init__(self, channels=[32, 64, 128]):
        super().__init__()
        # 下采样块：Conv(stride=2) + LeakyReLU
        self.conv1 = Conv2d(1, channels[0], kernel_size=3, stride=2, padding=1)  # 32→16
        self.lrelu1 = LeakyReLU(0.2)

        self.conv2 = Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)  # 16→8
        self.bn2 = BatchNorm2d(channels[1])
        self.lrelu2 = LeakyReLU(0.2)

        self.conv3 = Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)  # 8→4
        self.bn3 = BatchNorm2d(channels[2])
        self.lrelu3 = LeakyReLU(0.2)

        # 分类头
        self.flatten = Flatten()
        self.fc = Linear(channels[2] * 4 * 4, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.sigmoid(self.fc(x))  # (batch, 1)
        return F.reshape(x, (-1,))  # (batch,)


@register_model('dcgan_lite')
class DCGANLite(Module):
    """DCGAN 轻量版（图像生成）。

    组合 Generator 和 Discriminator，提供 GAN 训练的辅助方法。

    Args:
        z_dim: 噪声维度
        g_channels: 生成器通道列表
        d_channels: 判别器通道列表
    """

    def __init__(self, z_dim=64, g_channels=None, d_channels=None):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim, g_channels or [128, 64, 32])
        self.discriminator = Discriminator(d_channels or [32, 64, 128])

    def generate(self, batch_size, rng=None):
        """生成图像（推理用）。"""
        rng = rng or np.random.RandomState(42)
        z = Tensor(rng.randn(batch_size, self.z_dim).astype(np.float64))
        return self.generator(z)

    def forward(self, real_imgs):
        """前向传播（训练用，返回 D 的真假判断）。

        Returns:
            d_real: 判别器对真实图像的判断
            d_fake: 判别器对生成图像的判断
            fake_imgs: 生成的图像
        """
        batch = real_imgs.data.shape[0]
        # 生成假图像
        z = Tensor(np.random.randn(batch, self.z_dim).astype(np.float64))
        fake_imgs = self.generator(z)
        # 判别
        d_real = self.discriminator(real_imgs)
        d_fake = self.discriminator(fake_imgs)
        return d_real, d_fake, fake_imgs

"""Vision Transformer（ViT）轻量版：图像分类。

架构：图像切 patch → 线性投影 → +位置编码 → +<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token → Transformer Encoder → <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 输出 → 分类头。

轻量配置（默认）：32×32 图像、patch=4（64 个 patch）、d_model=64、4 头、2 层，参数量 ~80K，CPU 可训练。
"""
from __future__ import annotations

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from ..nn.module import Module, Parameter
from ..nn.layers import Linear
from ..nn.attention import TransformerEncoder
from .registry import register_model


@register_model('vit_lite')
class ViTLite(Module):
    """Vision Transformer 轻量版（图像分类）。

    Args:
        img_size: 输入图像边长（默认 32，假设正方形）
        patch_size: patch 边长（默认 4，img_size 必须能被 patch_size 整除）
        in_channels: 输入通道数（默认 3）
        d_model: Transformer 隐藏维度（默认 64）
        num_heads: 多头注意力头数（默认 4，d_model 必须能被 num_heads 整除）
        num_layers: Transformer 编码器层数（默认 2）
        d_ff: FFN 中间维度（默认 None → 4×d_model）
        num_classes: 分类类别数（默认 10）
        dropout: dropout 概率（当前未启用，预留接口）
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 d_model=64, num_heads=4, num_layers=2, d_ff=None,
                 num_classes=10, dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0, f'img_size({img_size}) 必须能被 patch_size({patch_size}) 整除'
        assert d_model % num_heads == 0, f'd_model({d_model}) 必须能被 num_heads({num_heads}) 整除'

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Patch Embedding：每个 patch 展平后线性投影到 d_model
        self.patch_embed = Linear(self.patch_dim, d_model)

        # <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token（可学习参数，拼接到 patch 序列最前面）
        self.cls_token = Parameter(np.zeros((1, 1, d_model), dtype=np.float64))
        # 可学习位置编码（<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> + num_patches 个位置）
        self.pos_embed = Parameter(np.zeros((1, self.num_patches + 1, d_model), dtype=np.float64))

        # Transformer Encoder
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)

        # 分类头：用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 的输出做分类
        self.head = Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        """初始化 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 和位置编码（小值正态分布，ViT 标准做法）。"""
        self.cls_token.data = np.random.randn(*self.cls_token.data.shape).astype(np.float64) * 0.02
        self.pos_embed.data = np.random.randn(*self.pos_embed.data.shape).astype(np.float64) * 0.02

    def _patchify(self, x):
        """把图像切成 patch 序列。

        x: (batch, C, H, W) → (batch, num_patches, patch_dim)
        """
        batch, C, H, W = x.data.shape
        P = self.patch_size
        n_h, n_w = H // P, W // P
        # (batch, C, n_h, P, n_w, P)
        x = F.reshape(x, (batch, C, n_h, P, n_w, P))
        # (batch, n_h, n_w, C, P, P) —— 把空间维提到通道前面
        x = F.transpose(x, axes=(0, 2, 4, 1, 3, 5))
        # (batch, num_patches, patch_dim)
        x = F.reshape(x, (batch, n_h * n_w, C * P * P))
        return x

    def forward(self, x):
        """前向传播。

        Args:
            x: (batch, in_channels, img_size, img_size)

        Returns:
            logits: (batch, num_classes)
        """
        batch = x.data.shape[0]

        # 1. Patch Embedding
        x = self._patchify(x)              # (batch, num_patches, patch_dim)
        x = self.patch_embed(x)             # (batch, num_patches, d_model)

        # 2. 拼接 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token
        cls = F.broadcast_to(self.cls_token, (batch, 1, self.d_model))
        x = F.concat([cls, x], dim=1)      # (batch, num_patches+1, d_model)

        # 3. 加位置编码（broadcast: (1, N+1, d_model) → (batch, N+1, d_model)）
        x = x + self.pos_embed

        # 4. Transformer Encoder
        x = self.encoder(x)                 # (batch, num_patches+1, d_model)

        # 5. 提取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 的输出（引擎不支持下标索引，用 narrow + reshape）
        cls_out = F.narrow(x, dim=1, start=0, length=1)  # (batch, 1, d_model)
        cls_out = F.reshape(cls_out, (batch, self.d_model))  # (batch, d_model)

        # 6. 分类头
        return self.head(cls_out)            # (batch, num_classes)

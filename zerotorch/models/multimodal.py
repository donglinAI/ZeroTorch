"""多模态融合模型：图文匹配（图像+文本 → 是否匹配）。

架构：图像编码器（轻量 CNN）+ 文本编码器（Embedding+MeanPooling）→ 融合层 → 二分类。
轻量配置：32×32 图像、vocab=500、64维图像特征+32维文本特征，参数量 ~80K。
"""
from __future__ import annotations

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from ..nn.module import Module
from ..nn.layers import Linear, Conv2d, BatchNorm2d, ReLU, Flatten, Embedding, Dropout
from .registry import register_model


class ImageEncoder(Module):
    """图像编码器：轻量 CNN，输出 64 维特征。"""

    def __init__(self, out_dim=64):
        super().__init__()
        self.conv1 = Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # 32→16
        self.bn1 = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16→8
        self.bn2 = BatchNorm2d(32)
        self.relu2 = ReLU()
        self.conv3 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 8→4
        self.bn3 = BatchNorm2d(64)
        self.relu3 = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(64 * 4 * 4, out_dim)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        return self.relu(self.fc(x))


class TextEncoder(Module):
    """文本编码器：Embedding + MeanPooling，输出 32 维特征。"""

    def __init__(self, vocab_size=500, embed_dim=32, out_dim=32):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.fc = Linear(embed_dim, out_dim)
        self.relu = ReLU()

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len) token id
        Returns:
            (batch, out_dim) 文本特征
        """
        emb = self.embedding(input_ids)  # (batch, seq, embed_dim)
        # MeanPooling：对 seq 维取平均
        pooled = F.mean(emb, dim=1)  # (batch, embed_dim)
        return self.relu(self.fc(pooled))


@register_model('multimodal_lite')
class MultimodalLite(Module):
    """多模态融合模型（图文匹配）。

    输入：图像 (batch, 1, 32, 32) + 文本 (batch, seq_len)
    输出：(batch, 2) 匹配/不匹配 logits

    Args:
        vocab_size: 词表大小
        img_dim: 图像特征维度
        txt_dim: 文本特征维度
        num_classes: 分类数（默认 2，匹配/不匹配）
    """

    def __init__(self, vocab_size=500, img_dim=64, txt_dim=32, num_classes=2):
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim=img_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=txt_dim, out_dim=txt_dim)
        # 融合层：concat 图像+文本特征 → MLP → 分类
        self.fc1 = Linear(img_dim + txt_dim, 64)
        self.relu = ReLU()
        self.dropout = Dropout(0.1)
        self.fc2 = Linear(64, num_classes)

    def forward(self, images, input_ids):
        """
        Args:
            images: (batch, 1, 32, 32)
            input_ids: (batch, seq_len)
        Returns:
            (batch, num_classes) 分类 logits
        """
        img_feat = self.image_encoder(images)  # (batch, img_dim)
        txt_feat = self.text_encoder(input_ids)  # (batch, txt_dim)
        fused = F.concat([img_feat, txt_feat], dim=1)  # (batch, img_dim+txt_dim)
        x = self.dropout(self.relu(self.fc1(fused)))
        return self.fc2(x)  # (batch, num_classes)

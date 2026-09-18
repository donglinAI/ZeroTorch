"""注意力机制：SelfAttention / MultiHeadAttention / TransformerBlock / TransformerEncoder。

全部用引擎算子手写前向，自动微分自动获得反向（与 RNN/LSTM 同理）。

输入约定：x 形状 (batch, seq_len, d_model)。
"""
from __future__ import annotations

import math

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from .module import Module
from .layers import Linear, LayerNorm, GELU
from .container import Sequential, ModuleList


def _swap_last_two(x):
    """交换张量最后两个维度（引擎 transpose 需要完整 axes，此辅助函数动态构造）。"""
    ndim = x.data.ndim
    axes = list(range(ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return F.transpose(x, axes=tuple(axes))


class SelfAttention(Module):
    """单头缩放点积自注意力（Scaled Dot-Product Self-Attention）。

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """

    def __init__(self, d_model, d_k=None):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.W_q = Linear(d_model, self.d_k, bias=False)
        self.W_k = Linear(d_model, self.d_k, bias=False)
        self.W_v = Linear(d_model, self.d_k, bias=False)

    def forward(self, x):
        # x: (batch, seq, d_model)
        Q = self.W_q(x)   # (batch, seq, d_k)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = F.matmul(Q, _swap_last_two(K)) / math.sqrt(self.d_k)  # (batch, seq, seq)
        attn = F.softmax(scores, dim=-1)
        out = F.matmul(attn, V)   # (batch, seq, d_k)
        return out


class MultiHeadAttention(Module):
    """多头自注意力（Multi-Head Self-Attention）。

    把 d_model 分成 num_heads 个 d_k，每头独立计算注意力后拼接，再做输出投影。
    """

    def __init__(self, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0, f'd_model({d_model}) 必须能被 num_heads({num_heads}) 整除'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 合并投影：一次 Linear 输出 d_model，后续 reshape 分头
        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.W_o = Linear(d_model, d_model, bias=False)

    def _split_heads(self, x):
        """(batch, seq, d_model) -> (batch, num_heads, seq, d_k)"""
        batch, seq, _ = x.data.shape
        x = F.reshape(x, (batch, seq, self.num_heads, self.d_k))
        # (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        return F.transpose(x, axes=(0, 2, 1, 3))

    def _merge_heads(self, x):
        """(batch, num_heads, seq, d_k) -> (batch, seq, d_model)"""
        # (batch, heads, seq, d_k) -> (batch, seq, heads, d_k)
        x = F.transpose(x, axes=(0, 2, 1, 3))
        batch, seq, _, _ = x.data.shape
        return F.reshape(x, (batch, seq, self.d_model))

    def forward(self, x, mask=None):
        batch, seq, _ = x.data.shape
        Q = self._split_heads(self.W_q(x))   # (batch, heads, seq, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))
        # 缩放点积注意力（批量矩阵乘自动覆盖 heads 维）
        scores = F.matmul(Q, _swap_last_two(K)) / math.sqrt(self.d_k)  # (batch, heads, seq, seq)
        if mask is not None:
            scores = scores + mask  # causal mask：上三角为 -inf，softmax 后未来权重为 0
        attn = F.softmax(scores, dim=-1)
        context = F.matmul(attn, V)   # (batch, heads, seq, d_k)
        context = self._merge_heads(context)   # (batch, seq, d_model)
        return self.W_o(context)


class TransformerBlock(Module):
    """Pre-LN Transformer 编码器块。

    x = x + MHA(LN(x))
    x = x + FFN(LN(x))
    """

    def __init__(self, d_model, num_heads=4, d_ff=None, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn = Sequential(
            Linear(d_model, self.d_ff),
            GELU(),
            Linear(self.d_ff, d_model),
        )

    def forward(self, x, mask=None):
        # Pre-LN：先归一化再注意力，残差连接
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(Module):
    """多层 Transformer 编码器堆叠 + 最终 LayerNorm。"""

    def __init__(self, d_model, num_heads=4, num_layers=2, d_ff=None, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)

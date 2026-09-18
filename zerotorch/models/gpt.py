"""GPT 轻量版（Decoder-only）：自回归文本生成。

GPT 核心：Decoder-only Transformer，causal mask 阻止注意到未来位置，逐词预测下一个词。
本教学版实现：词嵌入+位置嵌入 → Transformer Encoder（causal mask）→ LM Head → 逐词生成。

轻量配置：vocab=500、d_model=64、4头、2层、max_seq_len=32，参数量 ~150K，CPU 可训练。
"""
from __future__ import annotations

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from ..nn.module import Module
from ..nn.layers import Embedding, Linear, LayerNorm, Dropout
from ..nn.attention import TransformerEncoder
from .registry import register_model


def make_causal_mask(seq_len):
    """生成 causal mask（下三角掩码）。

    上三角（未来位置）为 -inf，下三角和对角线为 0。
    softmax 后未来位置的注意力权重为 0。

    Returns:
        numpy (1, 1, seq_len, seq_len)，可 broadcast 到 (batch, heads, seq, seq)
    """
    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float64), k=1) * -1e9
    return mask[None, None, :, :]  # (1, 1, seq, seq)


class GPTEmbedding(Module):
    """GPT 嵌入层：词嵌入 + 位置嵌入 + Dropout。"""

    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.word_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = Embedding(max_seq_len, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len) token id
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len = input_ids.shape
        word_emb = self.word_embeddings(input_ids)
        position_ids = Tensor(np.arange(seq_len, dtype=np.int64)[None, :].repeat(batch, axis=0))
        pos_emb = self.position_embeddings(position_ids)
        embeddings = word_emb + pos_emb
        return self.dropout(embeddings)


@register_model('gpt_lite')
class GPTLite(Module):
    """GPT 轻量版（Decoder-only，自回归文本生成）。

    Args:
        vocab_size: 词表大小
        d_model: 隐藏维度
        num_heads: 多头注意力头数
        num_layers: Transformer 层数
        max_seq_len: 最大序列长度
        d_ff: FFN 中间维度（默认 4×d_model）
        dropout: dropout 概率
    """

    def __init__(self, vocab_size=500, d_model=64, num_heads=4, num_layers=2,
                 max_seq_len=32, d_ff=None, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = GPTEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.ln_f = LayerNorm(d_model)  # 最终 LayerNorm（GPT 标准做法）
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, input_ids):
        """前向传播（训练用，teacher forcing）。

        Args:
            input_ids: (batch, seq_len) token id

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = input_ids.shape
        emb = self.embedding(input_ids)
        # causal mask：阻止注意到未来位置
        mask = Tensor(make_causal_mask(seq_len))
        hidden = self.encoder(emb, mask=mask)
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)

    def generate(self, start_ids, max_new_tokens=20, temperature=1.0):
        """自回归生成（推理用）。

        Args:
            start_ids: list[int] 起始 token id 序列
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（越高越随机）

        Returns:
            list[int] 完整 token id 序列（起始 + 生成）
        """
        self.eval()
        generated = list(start_ids)
        for _ in range(max_new_tokens):
            # 截断到最大长度
            input_seq = generated[-self.max_seq_len:]
            input_ids = Tensor(np.array([input_seq], dtype=np.int64))
            logits = self.forward(input_ids)
            # 取最后一个位置的 logits
            last_logits = logits.data[0, -1, :] / temperature
            # softmax 采样
            last_logits = last_logits - last_logits.max()
            exp = np.exp(last_logits)
            probs = exp / exp.sum()
            next_id = int(np.random.choice(len(probs), p=probs))
            generated.append(next_id)
        self.train()
        return generated

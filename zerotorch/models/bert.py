"""BERT 轻量版（Encoder-only）：MLM（Masked Language Model）预训练。

BERT 核心：双向 Transformer Encoder，通过 MLM 任务预训练，理解上下文。
本教学版实现：词嵌入+位置嵌入+Segment 嵌入 → Transformer Encoder → MLM Head → 预测被 mask 的词。

轻量配置：vocab=500、d_model=64、4头、2层、max_seq_len=32，参数量 ~150K，CPU 可训练。
"""
from __future__ import annotations

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from ..nn.module import Module
from ..nn.layers import Embedding, Linear, LayerNorm, GELU, Dropout
from ..nn.attention import TransformerEncoder
from .registry import register_model


class BERTEmbedding(Module):
    """BERT 嵌入层：词嵌入 + 位置嵌入 + Segment 嵌入 + LayerNorm + Dropout。"""

    def __init__(self, vocab_size, d_model, max_seq_len, num_segments=2, dropout=0.1):
        super().__init__()
        self.word_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = Embedding(max_seq_len, d_model)
        self.token_type_embeddings = Embedding(num_segments, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        """
        Args:
            input_ids: (batch, seq_len) token id
            token_type_ids: (batch, seq_len) segment id（0/1），默认全 0
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len = input_ids.shape
        # 词嵌入
        word_emb = self.word_embeddings(input_ids)
        # 位置嵌入：位置 0~seq_len-1
        position_ids = Tensor(np.arange(seq_len, dtype=np.int64)[None, :].repeat(batch, axis=0))
        pos_emb = self.position_embeddings(position_ids)
        # Segment 嵌入
        if token_type_ids is None:
            token_type_ids = Tensor(np.zeros((batch, seq_len), dtype=np.int64))
        token_type_emb = self.token_type_embeddings(token_type_ids)
        # 求和 + LayerNorm + Dropout
        embeddings = word_emb + pos_emb + token_type_emb
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class MLMHead(Module):
    """MLM 预测头：Linear → GELU → LayerNorm → Linear（输出词表 logits）。"""

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = Linear(d_model, d_model)
        self.act = GELU()
        self.layer_norm = LayerNorm(d_model)
        self.decoder = Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, vocab_size)
        """
        x = self.dense(x)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.decoder(x)


@register_model('bert_lite')
class BERTLite(Module):
    """BERT 轻量版（Encoder-only，MLM 预训练）。

    Args:
        vocab_size: 词表大小
        d_model: 隐藏维度
        num_heads: 多头注意力头数
        num_layers: Transformer 层数
        max_seq_len: 最大序列长度
        d_ff: FFN 中间维度（默认 4×d_model）
        dropout: dropout 概率
        num_segments: segment 类型数（默认 2，BERT 标准 A/B）
    """

    def __init__(self, vocab_size=500, d_model=64, num_heads=4, num_layers=2,
                 max_seq_len=32, d_ff=None, dropout=0.1, num_segments=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len,
                                       num_segments, dropout)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers,
                                          d_ff, dropout)
        self.mlm_head = MLMHead(d_model, vocab_size)

    def forward(self, input_ids, token_type_ids=None):
        """前向传播。

        Args:
            input_ids: (batch, seq_len) token id
            token_type_ids: (batch, seq_len) segment id（可选）

        Returns:
            mlm_logits: (batch, seq_len, vocab_size)
            sequence_output: (batch, seq_len, d_model)（encoder 输出，供下游任务用）
        """
        emb = self.embedding(input_ids, token_type_ids)
        seq_out = self.encoder(emb)
        mlm_logits = self.mlm_head(seq_out)
        return mlm_logits, seq_out


# ============================================================
# MLM 数据工具：随机 mask
# ============================================================
def create_mlm_mask(input_ids, vocab_size, mask_prob=0.15, mask_token_id=4):
    """对 token 序列随机 mask，返回 masked_input 和 mlm_labels。

    BERT MLM 策略：
    - 随机选 15% 的 token
    - 其中 80% 替换为 [MASK] token
    - 10% 替换为随机 token
    - 10% 保持不变
    - mlm_labels 只在被 mask 的位置有原 token id，其他位置为 -100（忽略索引）

    Args:
        input_ids: (batch, seq_len) numpy array
        vocab_size: 词表大小
        mask_prob: mask 概率（默认 0.15）
        mask_token_id: [MASK] token 的 id（默认 4，假设 0=PAD,1=UNK,2=CLS,3=SEP,4=MASK）

    Returns:
        masked_ids: (batch, seq_len) 被 mask 后的 token id
        mlm_labels: (batch, seq_len) MLM 标签（-100 表示忽略）
    """
    input_ids = np.asarray(input_ids)
    masked_ids = input_ids.copy()
    mlm_labels = np.full(input_ids.shape, -100, dtype=np.int64)

    mask = np.random.random(input_ids.shape) < mask_prob
    # 不 mask [CLS]/[SEP]/PAD（假设前 5 个是特殊 token）
    special = input_ids < 5
    mask = mask & ~special

    mlm_labels[mask] = input_ids[mask]

    # 80% 替换为 [MASK]
    mask_80 = mask & (np.random.random(input_ids.shape) < 0.8)
    masked_ids[mask_80] = mask_token_id

    # 10% 替换为随机 token
    mask_10 = mask & (np.random.random(input_ids.shape) < 0.5) & ~mask_80
    random_tokens = np.random.randint(5, vocab_size, size=input_ids.shape)
    masked_ids[mask_10] = random_tokens[mask_10]

    # 10% 保持不变（不做任何操作）
    return masked_ids, mlm_labels


def mlm_loss(mlm_logits, mlm_labels):
    """计算 MLM 损失（只在非 -100 位置计算 CrossEntropy）。

    全部用引擎算子计算，保留计算图，支持反向传播。

    Args:
        mlm_logits: Tensor (batch, seq_len, vocab_size)
        mlm_labels: numpy (batch, seq_len)，-100 表示忽略

    Returns:
        loss: Tensor 标量
    """
    batch, seq_len, vocab_size = mlm_logits.shape
    logits_2d = F.reshape(mlm_logits, (batch * seq_len, vocab_size))
    labels_1d = mlm_labels.reshape(-1)  # numpy
    mask = labels_1d != -100  # numpy bool

    # 把 -100 替换成 0（避免索引错误），后续用 mask 加权
    safe_labels = labels_1d.copy()
    safe_labels[~mask] = 0

    # one-hot 编码（numpy）
    one_hot = np.zeros((len(safe_labels), vocab_size), dtype=np.float64)
    one_hot[np.arange(len(safe_labels)), safe_labels] = 1.0

    # log softmax → 逐位置 NLL → mask 加权平均
    log_probs = F.log_softmax(logits_2d, dim=-1)           # (N, vocab)
    nll_per_pos = -F.sum(log_probs * Tensor(one_hot), dim=-1)  # (N,)
    mask_tensor = Tensor(mask.astype(np.float64))
    loss = F.sum(nll_per_pos * mask_tensor) / max(mask.sum(), 1)
    return loss

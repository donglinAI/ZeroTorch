# ZeroTorch

> 🧠 用纯 Python + NumPy 从零实现的 PyTorch 风格深度学习引擎

**不依赖任何深度学习框架，手写自动微分 + 47 算子 + 8 个经典模型，一行 import 即可训练。**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-≥1.24-green)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/donglinAI/ZeroTorch)

---

## ✨ 为什么做这个？

> *"我用了 7 年深度学习，但每次拆开 PyTorch 源码都觉得隔着一层黑盒。"*

我们往往通过微调开源模型、改造 loss 解决问题，但对**梯度怎么流的、算子怎么反传的、训练怎么收敛的**，始终缺少系统性理解。

ZeroTorch 就是要把这层窗户纸捅破——**从零手写一个深度学习引擎**，把自动微分、算子库、训练器、模型库、量化、推理部署全部跑通，每个算子的反向公式都显式可查。

---

## 🚀 核心特性

| 特性 | 实现 |
|---|---|
| 🔧 **自动微分** | 动态计算图 + DFS 拓扑排序 + 逆拓扑传播，47 算子全部通过数值梯度检查 |
| 🧩 **即插即用** | 4 类注册表（模型/损失/优化器/指标），新增组件不改框架代码 |
| 📦 **经典模型全覆盖** | 8 个轻量模型：分类/检测/分割/文本/生成/多模态 |
| ⚡ **PyTorch 风格 API** | `tensor() / nn.Module / loss / optim / DataLoader / Trainer` |
| 📉 **INT8 量化** | 4x 压缩，平均误差 < 1%，精度损失可忽略 |
| 🎯 **教学优先** | 纯 NumPy + CPU，无 GPU 依赖，每个算子独立成文件、反向公式可查 |

---

## 📦 模型库

| 模型 | 任务 | 参数量 | 效果 |
|---|---|---|---|
| **ViT-lite** | 图像分类 | 74K | acc=100% |
| **ResNet-lite** | 图像分类 | 78K | loss 0.69→0.025 |
| **YOLOv3-lite** | 目标检测 | 62K | loss 7.35→0.43 |
| **BERT-lite** | 文本理解（MLM） | 170K | loss 6.42→5.60 |
| **U-Net** | 图像分割 | 122K | loss 1.10→0.26 |
| **GPT-lite** | 文本生成 | 166K | loss 6.26→0.20 |
| **DCGAN-lite** | 图像生成 | 321K | 对抗收敛 |
| **Multimodal-lite** | 多模态图文匹配 | 112K | acc=100% |

---

## 🏃 快速开始

### 安装

```bash
git clone https://github.com/donglinAI/ZeroTorch.git
cd ZeroTorch
pip install numpy matplotlib
```

### 5 行代码训练一个分类器

```python
import zerotorch as zt
import numpy as np

# 1. 准备数据
X = np.random.randn(100, 1, 32, 32)
y = np.random.randint(0, 3, 100)
dl = zt.data.DataLoader(zt.data.TensorDataset(X, y), batch_size=16, shuffle=True)

# 2. 创建模型
model = zt.models.ResNetLite(in_channels=1, num_classes=3)

# 3. 训练
trainer = zt.Trainer(model, zt.loss.CrossEntropyLoss(),
                     zt.optim.Adam(model.parameters(), lr=1e-3),
                     dl, epochs=10, metrics=['accuracy'])
history = trainer.fit()
```

### 推理 + 量化

```python
from zerotorch.inference import InferenceEngine
from zerotorch.quantization import quantize_model

engine = InferenceEngine(model)
output = engine.predict(input_tensor)          # 推理
bench = engine.benchmark(input_tensor)         # 性能基准
q_params, _, _, ratio = quantize_model(model)  # INT8 量化 4x 压缩
```

---

## 📁 项目结构

```
zerotorch/
├── tensor.py            # 🧠 Tensor 自动微分核心
├── ops/                 # 🔧 47 算子库
├── nn/                  # 🧩 神经网络层（Linear/Conv/Attention/RNN...）
├── models/              # 📦 8 个经典模型（注册表即插即用）
├── loss/                # 📉 损失函数（7 种）
├── optim/               # ⚙️ 优化器（SGD/Adam/AdamW + 调度）
├── data/                # 📊 数据管线（Dataset/DataLoader/Transforms）
├── trainer.py           # 🏋️ 训练器（Trainer + Callback）
├── inference.py         # 🚀 推理引擎（predict + benchmark）
├── quantization.py      # 📉 INT8 量化压缩
├── parallel/            # 🔀 并行训练（教学版 DataParallel）
├── viz/                 # 📊 可视化（训练曲线/检测框/图像网格）
└── utils/               # 🛠️ 工具（gradcheck 数值梯度验证）
```

---

## 🧪 验证

```bash
python smoke_test_l1.py           # L1 自动微分（35 项）
python smoke_test_l4.py           # L4 全模型（28 项）
```

---

## 📚 文档

完整的源码精讲文档（逐函数拆解）见飞书：
[**ZeroTorch 架构设计文档**](https://feishu.doubao.com/docx/SpIwdAN5yodQOqx2iimcYdydnOd)

包含 32 个子文档，覆盖每个核心文件的逐函数精讲。

---

## 🎯 适用场景

- ✅ 想彻底搞懂深度学习底层原理的工程师
- ✅ 面试准备：手写自动微分/反向传播/算子
- ✅ 教学演示：从零实现一个深度学习框架
- ✅ 技术分享：拆解 PyTorch 内部机制

---

## ⚠️ 说明

- **纯 CPU + 纯 NumPy**，无 GPU 加速，性能不是目标
- **教学优先**，代码可读性高于执行效率
- **小规模模型**（70K-300K 参数），适合学习演示

---

## 📄 License

MIT License

---

<div align="center">
如果这个项目对你有帮助，欢迎 ⭐ Star 支持！
</div>

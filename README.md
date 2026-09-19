# ZeroTorch

> 🧠 用纯 Python + NumPy 从零实现的 PyTorch 风格深度学习引擎

**不依赖任何深度学习框架，手写自动微分 + 47 算子 + 3 个经典视觉模型，一行 import 即可训练。**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-≥1.24-green)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/donglinAI/ZeroTorch)

---

## ✨ 为什么做这个？

> *"我用了 7 年深度学习，但每次拆开 PyTorch 源码都觉得隔着一层黑盒。"*

我们往往通过微调开源模型、改造 loss 解决问题，但对**梯度怎么流的、算子怎么反传的、训练怎么收敛的**，始终缺少系统性理解。

ZeroTorch 就是要把这层窗户纸捅破——**从零手写一个深度学习引擎**，把自动微分、算子库、训练器、模型库全部跑通，每个算子的反向公式都显式可查。

---

## 🚀 核心特性

| 特性 | 实现 |
|---|---|
| 🔧 **自动微分** | 动态计算图 + DFS 拓扑排序 + 逆拓扑传播，47 算子全部通过数值梯度检查 |
| 🧩 **即插即用** | 4 类注册表（模型/损失/优化器/指标），新增组件不改框架代码 |
| 📦 **经典视觉模型** | 3 个轻量模型：分类（ResNet/ViT）+ 检测（YOLOv3） |
| ⚡ **PyTorch 风格 API** | `tensor() / nn.Module / loss / optim / DataLoader / Trainer` |
| 🎯 **教学优先** | 纯 NumPy + CPU，无 GPU 依赖，每个算子独立成文件、反向公式可查 |

---

## 📦 模型库

| 模型 | 任务 | 参数量 | 效果 |
|---|---|---|---|
| **ResNet-lite** | 图像分类（MNIST） | 177K | loss 1.19→0.28 |
| **ViT-lite** | 图像分类（MNIST） | 28K | acc 61.6%→ |
| **YOLOv3-lite** | 目标检测（circle+cross） | 62K | loss 3.03→0.06 |

---

## 🏃 快速开始

### 安装

```bash
git clone https://github.com/donglinAI/ZeroTorch.git
cd ZeroTorch
pip install numpy matplotlib
```

### 训练分类模型（ResNet）

```bash
python examples/train_resnet.py --epochs 10 --batch_size 4
```

### 训练 ViT 模型

```bash
python examples/train_vit.py --steps 100 --batch_size 1
```

### 训练 YOLO 检测模型

```bash
python examples/train_yolo.py --epochs 20 --batch_size 2
```

### 统一推理接口

```bash
# 推理 ResNet
python examples/inference.py --model resnet --weights resnet_weights.pt

# 推理 ViT
python examples/inference.py --model vit --weights vit_weights.pt

# 推理 YOLO
python examples/inference.py --model yolo --weights yolo_weights.pt
```

---

## 📁 项目结构

```
zerotorch/
├── tensor.py            # 🧠 Tensor 自动微分核心
├── ops/                 # 🔧 47 算子库
├── nn/                   # 🧩 神经网络层（Linear/Conv/Attention/RNN...）
├── models/               # 📦 3 个经典模型（注册表即插即用）
│   ├── resnet.py        #   ResNet-lite 图像分类
│   ├── yolo.py          #   YOLOv3-lite 目标检测
│   └── vit.py           #   ViT-lite 视觉 Transformer
├── loss/                 # 📉 损失函数（7 种）
├── optim/                # ⚙️ 优化器（SGD/Adam/AdamW + 调度）
├── data/                 # 📊 数据管线（Dataset/DataLoader/Transforms）
├── trainer.py            # 🏋️ 训练器（Trainer + Callback）
├── serialization.py     # 💾 模型保存/加载
└── utils/                # 🛠️ 工具（gradcheck 数值梯度验证）

examples/
├── train_resnet.py      # ResNet 分类训练
├── train_vit.py         # ViT 分类训练
├── train_yolo.py        # YOLO 检测训练
└── inference.py         # 统一推理接口
```

---

## 🧪 验证

```bash
python smoke_test_l1.py           # L1 自动微分（35 项）
python smoke_test_l2.py           # L2 神经网络层（28 项）
python smoke_test_l3.py           # L3 模型+训练器（20 项）
```

---

## 📚 文档

完整的源码精讲文档（逐函数拆解）见飞书：
[**ZeroTorch 架构设计文档**](https://feishu.doubao.com/docx/SpIwdAN5yodQOqx2iimcYdydnOd)

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
- **小规模模型**（28K-177K 参数），适合学习演示

---

## 📄 License

MIT License

---

<div align="center">
如果这个项目对你有帮助，欢迎 ⭐ Star 支持！
</div>

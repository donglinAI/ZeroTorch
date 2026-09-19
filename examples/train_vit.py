"""ViT-lite MNIST 手写数字分类训练脚本（轻量演示版）。

纯 NumPy 实现内存占用大，用 step 模式（每个 batch 只前向一次，省内存）。

用法：
    python examples/train_vit.py --steps 100
"""
import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import zerotorch as zt
from zerotorch.data.datasets import MNIST
from zerotorch.models import ViTLite


def main():
    parser = argparse.ArgumentParser(description='ViT-lite MNIST 手写数字分类训练（轻量演示版）')
    parser.add_argument('--data_root', type=str, default='./data', help='数据根目录')
    parser.add_argument('--steps', type=int, default=100, help='训练 step 数（每个 step 一次前向+反向）')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size（纯NumPy建议用1）')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--img_size', type=int, default=28, help='图像大小（MNIST 默认 28）')
    parser.add_argument('--patch_size', type=int, default=7, help='patch 大小（需整除 img_size）')
    parser.add_argument('--d_model', type=int, default=16, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=2, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=1, help='Transformer 层数')
    parser.add_argument('--max_train_samples', type=int, default=500, help='最大训练样本数')
    parser.add_argument('--save_path', type=str, default='vit_mnist_weights.pt', help='权重保存路径')
    args = parser.parse_args()

    print("=" * 60)
    print("ViT-lite MNIST 手写数字分类训练（轻量演示版）")
    print("=" * 60)

    # 1. 数据准备
    print("\n📊 数据准备（MNIST 手写数字）")
    train_ds = MNIST(root=args.data_root, train=True, download=True)

    X_train = train_ds.images[:args.max_train_samples].astype(np.float64)
    y_train = train_ds.labels[:args.max_train_samples].astype(np.int64)

    # 归一化到 [0, 1]
    X_train = X_train / 255.0

    # 调整形状到 (N, C, H, W)
    if X_train.ndim == 3:
        X_train = X_train[:, np.newaxis, :, :]

    print(f"   训练集: {len(X_train)} 张")
    print(f"   图像大小: {args.img_size}×{args.img_size}，10 类（0-9）")
    print(f"   patch: {args.patch_size}×{args.patch_size}（{(args.img_size//args.patch_size)**2} 个 patch）")
    print(f"   batch size: {args.batch_size}")

    # 2. 创建模型
    print("\n🏗️ 模型创建")
    model = ViTLite(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=1,
        num_classes=10,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    print(f"   参数量: {model.num_parameters():,}")

    # 3. 训练配置
    loss_fn = zt.loss.CrossEntropyLoss()
    optimizer = zt.optim.Adam(model.parameters(), lr=args.lr)

    # 4. 训练（step 模式：每个 batch 只前向一次，省内存）
    print(f"\n🏋️ 开始训练（{args.steps} 个 step）")
    model.train()

    for step in range(args.steps):
        # 随机取一个 batch
        idx = np.random.randint(0, len(X_train) - args.batch_size)
        x_batch = zt.tensor(X_train[idx:idx+args.batch_size])
        y_batch = zt.tensor(y_train[idx:idx+args.batch_size])

        # 前向 + 反向（只前向一次，省内存）
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印进度
        pred_cls = pred.data.argmax(axis=1)
        acc = (pred_cls == y_batch.data).mean()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"   step {step+1}/{args.steps} | loss={float(loss.data):.4f} | acc={acc:.2f}")

    # 5. 保存
    print(f"\n💾 保存模型: {args.save_path}")
    zt.serialization.save(model, args.save_path)

    # 6. 结果
    print(f"\n✅ 训练完成！")
    print(f"   最后 loss: {float(loss.data):.4f}")
    print(f"   最后 acc: {acc:.4f}")


if __name__ == '__main__':
    main()

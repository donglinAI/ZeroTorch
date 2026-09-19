"""ResNet-lite MNIST 手写数字分类训练脚本。

用法：
    python examples/train_resnet.py --epochs 5 --batch_size 4
"""
import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import zerotorch as zt
from zerotorch.data import DataLoader, TensorDataset
from zerotorch.data.datasets import MNIST
from zerotorch.models import ResNetLite


def main():
    parser = argparse.ArgumentParser(description='ResNet-lite MNIST 手写数字分类训练')
    parser.add_argument('--data_root', type=str, default='./data', help='数据根目录')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size（纯NumPy建议用小值）')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--img_size', type=int, default=28, help='图像大小（MNIST 默认 28）')
    parser.add_argument('--max_train_samples', type=int, default=2000, help='最大训练样本数（控制内存）')
    parser.add_argument('--max_val_samples', type=int, default=500, help='最大验证样本数')
    parser.add_argument('--save_path', type=str, default='resnet_mnist_weights.pt', help='权重保存路径')
    args = parser.parse_args()

    print("=" * 60)
    print("ResNet-lite MNIST 手写数字分类训练")
    print("=" * 60)

    # 1. 数据准备
    print("\n📊 数据准备（MNIST 手写数字）")
    train_ds = MNIST(root=args.data_root, train=True, download=True)
    val_ds = MNIST(root=args.data_root, train=False, download=True)

    # 取部分样本（控制内存，纯NumPy训练全量会OOM）
    X_train = train_ds.images[:args.max_train_samples].astype(np.float64)
    y_train = train_ds.labels[:args.max_train_samples].astype(np.int64)
    X_val = val_ds.images[:args.max_val_samples].astype(np.float64)
    y_val = val_ds.labels[:args.max_val_samples].astype(np.int64)

    # 调整形状到 (N, C, H, W)
    if X_train.ndim == 3:
        X_train = X_train[:, np.newaxis, :, :]
        X_val = X_val[:, np.newaxis, :, :]

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)

    print(f"   训练集: {len(X_train)} 张，{len(train_dl)} batches")
    print(f"   测试集: {len(X_val)} 张，{len(val_dl)} batches")
    print(f"   图像大小: {args.img_size}×{args.img_size}，10 类（0-9）")
    print(f"   batch size: {args.batch_size}（纯NumPy建议小值，避免OOM）")

    # 2. 创建模型
    print("\n🏗️ 模型创建")
    model = ResNetLite(
        in_channels=1,
        num_classes=10,
        channels=[8, 16, 32, 64],
        img_size=args.img_size
    )
    print(f"   参数量: {model.num_parameters():,}")

    # 3. 训练配置
    loss_fn = zt.loss.CrossEntropyLoss()
    optimizer = zt.optim.Adam(model.parameters(), lr=args.lr)

    # 4. 训练
    print("\n🏋️ 开始训练")
    trainer = zt.Trainer(
        model, loss_fn, optimizer,
        train_loader=train_dl,
        val_loader=val_dl,
        epochs=args.epochs,
        metrics=['accuracy'],
        callbacks=[zt.History(), zt.ProgressBar(),
                   zt.ModelCheckpoint(args.save_path, monitor='val_accuracy', mode='max')]
    )
    history = trainer.fit()

    # 5. 保存
    print(f"\n💾 保存模型: {args.save_path}")
    zt.serialization.save(model, args.save_path)

    # 6. 结果
    print(f"\n✅ 训练完成！")
    print(f"   最佳测试准确率: {max(history['val_accuracy']):.4f}")
    print(f"   最终训练 loss: {history['loss'][-1]:.4f}")


if __name__ == '__main__':
    main()

"""YOLOv3-lite 目标检测训练脚本。

用法：
    python examples/train_yolo.py --epochs 10 --batch_size 2 --lr 1e-3
"""
import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import zerotorch as zt
from zerotorch.data import DataLoader
from zerotorch.data.datasets import ShapesDetection
from zerotorch.models import YOLOv3Lite


def visualize_samples(ds, num_samples=4, save_path='yolo_samples.png'):
    """可视化检测数据集样本（图像 + 标注框）。"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    class_names = ['circle', 'square', 'triangle']
    colors = ['r', 'g', 'b']

    for i in range(num_samples):
        img, (boxes, labels) = ds[i]

        if img.ndim == 3:
            img_show = img[0]
        else:
            img_show = img

        axes[i].imshow(img_show, cmap='gray')

        S = ds.img_size
        for j in range(len(boxes)):
            cx, cy, w, h = boxes[j]
            cls = labels[j]
            x1 = (cx - w/2) * S
            y1 = (cy - h/2) * S
            rect = patches.Rectangle(
                (x1, y1), w*S, h*S,
                linewidth=2, edgecolor=colors[int(cls)], facecolor='none'
            )
            axes[i].add_patch(rect)
            axes[i].text(x1, y1-2, class_names[int(cls)],
                        color=colors[int(cls)], fontsize=8)

        axes[i].set_title(f'Ground Truth {i+1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"   真实标注可视化已保存: {save_path}")
    plt.close()


def visualize_predictions(model, ds, num_samples=4, save_path='yolo_predictions.png'):
    """可视化检测预测结果（和真实标注对比）。"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    class_names = ['circle', 'square', 'triangle']
    colors = ['r', 'g', 'b']
    S = ds.img_size

    fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))

    model.eval()

    for i in range(num_samples):
        img, (gt_boxes, gt_labels) = ds[i]

        img_batch = zt.tensor(img[np.newaxis, ...].astype(np.float64))
        pred = model(img_batch)

        pred_results = model.decode(pred.data, conf_threshold=0.3, img_size=S)
        pred_boxes = pred_results[0]['boxes']
        pred_labels = pred_results[0]['labels']
        pred_scores = pred_results[0]['scores']

        # 上排：真实标注
        axes[0, i].imshow(img[0], cmap='gray')
        for j in range(len(gt_boxes)):
            cx, cy, w, h = gt_boxes[j]
            cls = gt_labels[j]
            x1 = (cx - w/2) * S
            y1 = (cy - h/2) * S
            rect = patches.Rectangle(
                (x1, y1), w*S, h*S,
                linewidth=2, edgecolor=colors[int(cls)], facecolor='none'
            )
            axes[0, i].add_patch(rect)
        axes[0, i].set_title(f'GT {i+1}')
        axes[0, i].axis('off')

        # 下排：预测结果
        axes[1, i].imshow(img[0], cmap='gray')
        for j in range(len(pred_boxes)):
            cx, cy, w, h = pred_boxes[j]
            cls = pred_labels[j]
            score = pred_scores[j]
            x1 = (cx - w/2) * S
            y1 = (cy - h/2) * S
            rect = patches.Rectangle(
                (x1, y1), w*S, h*S,
                linewidth=2, edgecolor=colors[int(cls)], facecolor='none', linestyle='--'
            )
            axes[1, i].add_patch(rect)
            axes[1, i].text(x1, y1-2, f'{class_names[int(cls)]}:{score:.2f}',
                           color=colors[int(cls)], fontsize=7)
        axes[1, i].set_title(f'Pred {i+1} ({len(pred_boxes)} boxes)')
        axes[1, i].axis('off')

    plt.suptitle('YOLOv3-lite 检测结果（上排：真实标注，下排：预测）', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"   预测结果可视化已保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='YOLOv3-lite 目标检测训练')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--num_samples', type=int, default=200, help='样本数')
    parser.add_argument('--img_size', type=int, default=32, help='图像大小')
    parser.add_argument('--num_classes', type=int, default=3, help='类别数')
    parser.add_argument('--no_viz', action='store_true', help='不显示样本可视化')
    parser.add_argument('--save_path', type=str, default='yolov3_lite_weights.pt', help='权重保存路径')
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv3-lite 目标检测训练")
    print("=" * 60)

    # 1. 数据准备
    print("\n📊 数据准备（合成检测数据）")
    ds = ShapesDetection(num_samples=args.num_samples, img_size=args.img_size, max_objects=3)
    train_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    print(f"   样本数: {len(ds)}")
    print(f"   图像大小: {args.img_size}×{args.img_size}")
    print(f"   类别: circle(红) / square(绿) / triangle(蓝)")

    if not args.no_viz:
        print("\n🖼️  样本可视化（真实标注）")
        visualize_samples(ds, num_samples=4, save_path='yolo_samples.png')

    # 2. 创建模型
    print("\n🏗️ 模型创建")
    model = YOLOv3Lite(
        in_channels=1,
        num_classes=args.num_classes,
        anchors=[[10,10],[20,20],[40,40]],
        grid_size=8
    )
    print(f"   参数量: {model.num_parameters():,}")

    # 3. 训练配置
    loss_fn = model.make_loss()
    optimizer = zt.optim.Adam(model.parameters(), lr=args.lr)

    # 4. 训练（真正的 epoch 训练）
    print(f"\n🏋️ 开始训练（{args.epochs} 个 epoch）")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (xb, yb) in enumerate(train_dl):
            optimizer.zero_grad()
            pred = model(xb)

            # yb = (boxes, labels)，boxes shape (B, max_obj, 4)，labels shape (B, max_obj)
            # 拆成 list 传给 YOLOLoss
            gt_boxes_list = [yb[0].data[i] for i in range(yb[0].data.shape[0])]
            gt_labels_list = [yb[1].data[i] for i in range(yb[1].data.shape[0])]

            loss = loss_fn(pred, gt_boxes_list, gt_labels_list)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.data)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"   epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f}")

    # 5. 保存
    print(f"\n💾 保存模型: {args.save_path}")
    zt.serialization.save(model, args.save_path)

    # 6. 测试预测
    if not args.no_viz:
        print("\n🔍 测试预测结果")
        visualize_predictions(model, ds, num_samples=4, save_path='yolo_predictions.png')

    print(f"\n✅ 训练完成！")


if __name__ == '__main__':
    main()

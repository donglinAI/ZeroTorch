"""可视化：训练曲线 / 分类预测 / 检测框 / 图像网格（matplotlib Agg 后端）。"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def plot_curves(history, save_path, metrics=None):
    """绘制 loss（与可选指标）随 epoch 变化的曲线。"""
    metrics = metrics or []
    nrows = 1 + (1 if metrics else 0)
    fig, axes = plt.subplots(1, nrows, figsize=(5 * nrows, 4))
    axes = np.atleast_1d(axes)
    epochs = list(range(1, len(history.get('loss', [])) + 1))
    ax = axes[0]
    ax.plot(epochs, history.get('loss', []), marker='o', label='train loss')
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], marker='s', label='val loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    for i, name in enumerate(metrics, start=1):
        ax = axes[i]
        ax.plot(epochs, history.get(name, []), marker='o', label=f'train {name}')
        if f'val_{name}' in history and history[f'val_{name}']:
            ax.plot(epochs, history[f'val_{name}'], marker='s', label=f'val {name}')
        ax.set_xlabel('epoch')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=130)
    plt.close(fig)


def plot_classification(images, preds, labels, save_path, class_names=None,
                        n=10):
    """展示分类预测：每张图标注 '预测/真实'。images: (N, C, H, W)。"""
    n = min(n, len(images))
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        img = images[i]
        img = img.squeeze()
        axes[i].imshow(img, cmap='gray')
        p, t = int(preds[i]), int(labels[i])
        if class_names is not None:
            p, t = class_names[p], class_names[t]
        color = 'green' if p == t else 'red'
        axes[i].set_title(f'pred={p}\ntrue={t}', color=color, fontsize=9)
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=130)
    plt.close(fig)


def plot_detections(images, boxes, labels, save_path, class_names=None, n=3):
    """在图上绘制检测框（boxes 为 cxcywh 归一化坐标）。"""
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    axes = np.atleast_1d(axes)
    class_names = class_names or [str(i) for i in range(9)]
    for i in range(n):
        img = images[i].squeeze()
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        H, W = img.shape
        for k, box in enumerate(boxes[i]):
            cx, cy, w, h = box
            if w <= 1e-6:
                continue
            x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
            rect = plt.Rectangle((x0, y0), w * W, h * H,
                                 fill=False, edgecolor='lime', linewidth=1.5)
            ax.add_patch(rect)
            if labels is not None:
                cls = int(labels[i][k])
                ax.text(x0, max(y0 - 2, 0), class_names[cls % len(class_names)],
                        color='yellow', fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=130)
    plt.close(fig)


def plot_image_grid(images, save_path, nrow=5, titles=None, cmap='gray'):
    """图像网格展示（VAE 生成结果等）。images: (N, C, H, W) 或 (N, H, W)。"""
    images = np.asarray(images)
    n = len(images)
    cols = min(nrow, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        img = images[i].squeeze()
        axes[i].imshow(img, cmap=cmap)
        if titles is not None:
            axes[i].set_title(str(titles[i]), fontsize=8)
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=130)
    plt.close(fig)


def plot_text_history(losses, save_path):
    """字符模型训练 loss 曲线。"""
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Char-LSTM Training Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_ensure_dir(save_path), dpi=130)
    plt.close('all')

"""合成数据集：blobs / moons / 形状检测 。

合成数据让四个任务示例可以完全离线、可复现地跑通。
"""
from __future__ import annotations

import os

import numpy as np

from ..dataset import Dataset


# ---------------------------------------------------------------------------
# 经典玩具分类数据
# ---------------------------------------------------------------------------
def make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=1.0,
               center_box=(-6.0, 6.0), seed=0):
    rng = np.random.RandomState(seed)
    center = rng.uniform(center_box[0], center_box[1],
                         size=(centers, n_features))
    X, y = [], []
    per = n_samples // centers
    for c in range(centers):
        X.append(rng.randn(per, n_features) * cluster_std + center[c])
        y.append(np.full(per, c))
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    return X, y


def make_moons(n_samples=200, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    t = np.linspace(0, np.pi, n)
    x1 = np.stack([np.cos(t), np.sin(t)], axis=1)
    x2 = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
    X = (X + rng.randn(n_samples, 2) * noise).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# 形状检测：合成图片 + 目标框 + 类别
# ---------------------------------------------------------------------------
class ShapesDetection(Dataset):
    """生成包含圆形 / 方形 / 三角形的 32x32 灰度图，每张固定 3 个目标。

    返回: (img (1,32,32) float32, boxes (3,4) cxcywh 归一化, labels (3,) int64)
    """

    def __init__(self, num_samples=2000, img_size=32, max_objects=3,
                 num_classes=3, seed=0):
        self.num_samples = num_samples
        self.img_size = img_size
        self.max_objects = max_objects
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)
        self.samples = [self._make_one() for _ in range(num_samples)]

    def _make_one(self):
        S = self.img_size
        img = np.zeros((1, S, S), dtype=np.float32)
        boxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        labels = np.zeros(self.max_objects, dtype=np.int64)
        yy, xx = np.mgrid[0:S, 0:S]
        for k in range(self.max_objects):
            cls = self.rng.randint(0, self.num_classes)
            cx = self.rng.uniform(0.25, 0.75) * S
            cy = self.rng.uniform(0.25, 0.75) * S
            r = self.rng.uniform(2.0, 5.0)
            val = self.rng.uniform(0.3, 1.0)
            if cls == 0:                       # 圆
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            elif cls == 1:                     # 方
                mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
            else:                              # 三角
                v0 = np.array([cx, cy - r])
                v1 = np.array([cx - r * 0.9, cy + r * 0.7])
                v2 = np.array([cx + r * 0.9, cy + r * 0.7])
                mask = _in_triangle(xx, yy, v0, v1, v2)
            img[0, mask] = val
            boxes[k] = [cx / S, cy / S, (2 * r + 1) / S, (2 * r + 1) / S]
            labels[k] = cls
        # 按 x 中心排序，让输出槽位有确定语义（槽 0=最左……），消除对称歧义
        order = np.argsort(boxes[:, 0])
        return img, (boxes[order], labels[order])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.samples[index]


def _in_triangle(xx, yy, v0, v1, v2):
    def sign(a, b, c):
        return (a[..., 0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[..., 1] - c[1])
    d1 = sign(np.stack([xx, yy], axis=-1), v1, v0)
    d2 = sign(np.stack([xx, yy], axis=-1), v2, v1)
    d3 = sign(np.stack([xx, yy], axis=-1), v0, v2)
    neg = (d1 < 0) & (d2 < 0) & (d3 < 0)
    pos = (d1 > 0) & (d2 > 0) & (d3 > 0)
    return neg | pos

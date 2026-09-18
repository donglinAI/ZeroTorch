"""指标库：accuracy / IoU 等，注册机制便于扩展。"""
from __future__ import annotations

import numpy as np

METRIC_REGISTRY = {}


def register_metric(name):
    def deco(fn):
        fn.name = name
        METRIC_REGISTRY[name] = fn
        return fn
    return deco


@register_metric('accuracy')
def accuracy(logits, targets):
    """分类准确率。logits: (N, C) 或 (..., C)；targets: (...,)。"""
    preds = np.argmax(np.asarray(logits), axis=-1)
    return float(np.mean(preds == np.asarray(targets)))


def _xywh_to_xyxy(boxes):
    """(..., 4) 的 cxcywh 归一化框 -> (..., 4) 的 xyxy。"""
    boxes = np.asarray(boxes)
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def iou(box1, box2):
    """两个 (4,) 框（cxcywh）的 IoU。"""
    b1 = _xywh_to_xyxy(box1)
    b2 = _xywh_to_xyxy(box2)
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


@register_metric('mean_iou')
def mean_iou(pred_boxes, gt_boxes):
    """检测框质量：按 x 中心排序后逐位配对计算 IoU 并取平均。

    pred_boxes / gt_boxes: (N, K, 4) 的 cxcywh 归一化框。
    """
    pred = np.asarray(pred_boxes)
    gt = np.asarray(gt_boxes)
    N, K, _ = pred.shape
    total, cnt = 0.0, 0
    for i in range(N):
        p = pred[i]
        g = gt[i]
        # 按 x 中心排序做简单匹配（演示用途；工程上可用匈牙利算法）
        p = p[np.argsort(p[:, 0])]
        g = g[np.argsort(g[:, 0])]
        for k in range(K):
            if g[k, 2] > 1e-6 and g[k, 3] > 1e-6:    # 只统计真实存在的框
                total += iou(p[k], g[k])
                cnt += 1
    return total / cnt if cnt else 0.0

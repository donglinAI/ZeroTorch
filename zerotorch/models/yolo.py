"""YOLOv3 轻量版（教学简化版）：目标检测。

简化策略（对比原版 YOLOv3）：
- 单尺度检测（8×8 特征图），不做多尺度 FPN
- 3 个 anchor box
- 轻量 backbone（4 个卷积块），参数量 ~65K
- 输入 32×32 灰度图（适配 ShapesDetection 合成数据）

包含：YOLOv3Lite 模型 + YOLOLoss 损失 + 推理解码 + NMS 后处理。
"""
from __future__ import annotations

import numpy as np

from .. import functions as F
from ..tensor import Tensor
from ..nn.module import Module
from ..nn.container import Sequential
from ..nn.layers import Conv2d, BatchNorm2d, LeakyReLU
from .registry import register_model


# ============================================================
# Anchor 定义（3 个先验框，宽高以特征图像素为单位）
# ============================================================
DEFAULT_ANCHORS = np.array([
    [1.0, 1.0],   # 小方框
    [1.5, 1.5],   # 中方框
    [2.0, 2.0],   # 大方框
], dtype=np.float64)


# ============================================================
# 工具函数
# ============================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def compute_iou(boxes1, boxes2):
    """计算两组边界框的 IoU。

    boxes: (N, 4) 格式 [cx, cy, w, h]（归一化或像素均可，只要一致）
    返回: (N, M) IoU 矩阵
    """
    # 转成 [x1, y1, x2, y2]
    b1 = np.stack([
        boxes1[:, 0] - boxes1[:, 2] / 2,
        boxes1[:, 1] - boxes1[:, 3] / 2,
        boxes1[:, 0] + boxes1[:, 2] / 2,
        boxes1[:, 1] + boxes1[:, 3] / 2,
    ], axis=1)
    b2 = np.stack([
        boxes2[:, 0] - boxes2[:, 2] / 2,
        boxes2[:, 1] - boxes2[:, 3] / 2,
        boxes2[:, 0] + boxes2[:, 2] / 2,
        boxes2[:, 1] + boxes2[:, 3] / 2,
    ], axis=1)

    inter_x1 = np.maximum(b1[:, 0:1], b2[:, 0:1].T)
    inter_y1 = np.maximum(b1[:, 1:2], b2[:, 1:2].T)
    inter_x2 = np.minimum(b1[:, 2:3], b2[:, 2:3].T)
    inter_y2 = np.minimum(b1[:, 3:4], b2[:, 3:4].T)

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制（NMS）。

    boxes: (N, 4) [cx, cy, w, h]
    scores: (N,) 置信度
    返回: 保留的框索引
    """
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = compute_iou(boxes[i:i+1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_threshold]
    return keep


# ============================================================
# 目标分配：把 GT 框分配到网格 + anchor
# ============================================================
def build_targets(gt_boxes, gt_labels, grid_size, anchors, num_classes):
    """为一个样本构建 YOLO 训练目标。

    Args:
        gt_boxes: (N, 4) 归一化 [cx, cy, w, h]
        gt_labels: (N,) 类别标签
        grid_size: 特征图边长（如 8）
        anchors: (num_anchors, 2) anchor 宽高（特征图像素单位）
        num_classes: 类别数

    Returns:
        target_obj: (num_anchors, grid, grid) 置信度目标（0/1）
        target_box: (num_anchors, grid, grid, 4) 坐标目标 (tx, ty, tw, th)
        target_cls: (num_anchors, grid, grid, num_classes) 分类目标（one-hot）
        obj_mask: (num_anchors, grid, grid) 正样本掩码
    """
    num_anchors = len(anchors)
    target_obj = np.zeros((num_anchors, grid_size, grid_size), dtype=np.float64)
    target_box = np.zeros((num_anchors, grid_size, grid_size, 4), dtype=np.float64)
    target_cls = np.zeros((num_anchors, grid_size, grid_size, num_classes), dtype=np.float64)
    obj_mask = np.zeros((num_anchors, grid_size, grid_size), dtype=np.float64)

    for i in range(len(gt_boxes)):
        cx, cy, w, h = gt_boxes[i]
        label = int(gt_labels[i])
        # 中心点所在网格
        gi = int(cx * grid_size)
        gj = int(cy * grid_size)
        gi = min(gi, grid_size - 1)
        gj = min(gj, grid_size - 1)

        # 选择与 GT 框 IoU 最大的 anchor（只用宽高比计算）
        gt_wh = np.array([[w * grid_size, h * grid_size]])  # 转成特征图像素
        anchor_ious = compute_iou(
            np.concatenate([np.zeros((1, 2)), gt_wh], axis=1),
            np.concatenate([np.zeros((num_anchors, 2)), anchors], axis=1)
        )[0]
        best_anchor = np.argmax(anchor_ious)

        # 编码目标
        tx = cx * grid_size - gi
        ty = cy * grid_size - gj
        tw = np.log(max(w * grid_size / anchors[best_anchor, 0], 1e-6))
        th = np.log(max(h * grid_size / anchors[best_anchor, 1], 1e-6))

        obj_mask[best_anchor, gj, gi] = 1.0
        target_obj[best_anchor, gj, gi] = 1.0
        target_box[best_anchor, gj, gi] = [tx, ty, tw, th]
        target_cls[best_anchor, gj, gi, label] = 1.0

    return target_obj, target_box, target_cls, obj_mask


# ============================================================
# YOLO 损失函数
# ============================================================
class YOLOLoss:
    """YOLO 损失：坐标 MSE + 置信度 BCE + 分类 CrossEntropy。

    正样本：计算全部三部分损失
    负样本：只计算置信度损失（权重降低，避免正负样本不平衡）
    """

    def __init__(self, anchors, grid_size, num_classes,
                 coord_weight=5.0, obj_weight=1.0, noobj_weight=0.5, cls_weight=1.0):
        self.anchors = np.asarray(anchors, dtype=np.float64)
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.coord_weight = coord_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.cls_weight = cls_weight
        self.name = 'yolo_loss'

    def __call__(self, predictions, gt_boxes_list, gt_labels_list):
        """计算损失。

        Args:
            predictions: Tensor (batch, num_anchors*(5+num_classes), grid, grid)
            gt_boxes_list: list of (N_i, 4) 归一化 [cx, cy, w, h]
            gt_labels_list: list of (N_i,) 类别标签

        Returns:
            loss: Tensor 标量
        """
        batch = predictions.data.shape[0]
        num_anchors = len(self.anchors)
        grid = self.grid_size
        bbox_attrs = 5 + self.num_classes

        # reshape: (batch, A*C, G, G) -> (batch, A, G, G, C)
        pred = F.reshape(predictions, (batch, num_anchors, bbox_attrs, grid, grid))
        pred = F.transpose(pred, axes=(0, 1, 3, 4, 2))  # (batch, A, G, G, C)

        # 构建目标（numpy）
        t_obj = np.zeros((batch, num_anchors, grid, grid), dtype=np.float64)
        t_box = np.zeros((batch, num_anchors, grid, grid, 4), dtype=np.float64)
        t_cls = np.zeros((batch, num_anchors, grid, grid, self.num_classes), dtype=np.float64)
        mask = np.zeros((batch, num_anchors, grid, grid), dtype=np.float64)

        for b in range(batch):
            to, tb, tc, om = build_targets(
                gt_boxes_list[b], gt_labels_list[b], grid, self.anchors, self.num_classes)
            t_obj[b], t_box[b], t_cls[b], mask[b] = to, tb, tc, om

        # 提取预测分量（Tensor 不支持下标索引，用 F.narrow）
        pred_xy = F.narrow(pred, dim=4, start=0, length=2)    # (batch, A, G, G, 2)
        pred_wh = F.narrow(pred, dim=4, start=2, length=2)    # (batch, A, G, G, 2)
        pred_obj_raw = F.narrow(pred, dim=4, start=4, length=1)  # (batch, A, G, G, 1)
        pred_obj = F.reshape(pred_obj_raw, (batch, num_anchors, grid, grid))  # (batch, A, G, G)
        pred_cls = F.narrow(pred, dim=4, start=5, length=self.num_classes)  # (batch, A, G, G, num_classes)

        # 1. 坐标损失（仅正样本，MSE）
        mask_4d = mask.reshape(batch, num_anchors, grid, grid, 1)  # numpy (batch, A, G, G, 1)
        t_xy = t_box[:, :, :, :, 0:2]   # numpy 下标可用
        t_wh = t_box[:, :, :, :, 2:4]
        box_diff = (pred_xy - Tensor(t_xy)) ** 2 + (pred_wh - Tensor(t_wh)) ** 2
        coord_loss = F.sum(box_diff * Tensor(mask_4d)) / max(np.sum(mask), 1)

        # 2. 置信度损失（BCE with logits）
        obj_diff = F.binary_cross_entropy_with_logits(pred_obj, Tensor(t_obj)) * Tensor(mask)
        obj_loss = F.sum(obj_diff) / max(np.sum(mask), 1)
        noobj_mask = 1.0 - mask
        noobj_diff = F.binary_cross_entropy_with_logits(pred_obj, Tensor(t_obj)) * Tensor(noobj_mask)
        noobj_loss = F.sum(noobj_diff) / max(np.sum(noobj_mask), 1)

        # 3. 分类损失（仅正样本，sigmoid + BCE per class）
        cls_diff = F.binary_cross_entropy_with_logits(pred_cls, Tensor(t_cls)) * Tensor(mask_4d)
        cls_loss = F.sum(cls_diff) / max(np.sum(mask), 1)

        total = (self.coord_weight * coord_loss +
                 self.obj_weight * obj_loss +
                 self.noobj_weight * noobj_loss +
                 self.cls_weight * cls_loss)
        return total


# ============================================================
# 推理解码：网络输出 -> 边界框
# ============================================================
def decode_predictions(predictions, anchors, grid_size, num_classes,
                       conf_threshold=0.5, img_size=32):
    """把网络输出解码成边界框列表。

    Args:
        predictions: numpy (batch, num_anchors*(5+num_classes), grid, grid) 或单样本
        anchors: (num_anchors, 2)
        conf_threshold: 置信度阈值
        img_size: 原始图像边长（用于归一化）

    Returns:
        list of dicts: [{'boxes': (N,4) [cx,cy,w,h] 归一化, 'scores': (N,), 'labels': (N,)}]
    """
    if predictions.ndim == 3:
        predictions = predictions[None]
    batch = predictions.shape[0]
    num_anchors = len(anchors)
    bbox_attrs = 5 + num_classes

    results = []
    for b in range(batch):
        # (A*C, G, G) -> (A, G, G, C)
        pred = predictions[b].reshape(num_anchors, bbox_attrs, grid_size, grid_size)
        pred = pred.transpose(0, 2, 3, 1)  # (A, G, G, C)

        boxes = []
        scores = []
        labels = []
        for a in range(num_anchors):
            for gj in range(grid_size):
                for gi in range(grid_size):
                    raw = pred[a, gj, gi]
                    obj = _sigmoid(raw[4])
                    if obj < conf_threshold:
                        continue
                    cx = (gi + _sigmoid(raw[0])) / grid_size
                    cy = (gj + _sigmoid(raw[1])) / grid_size
                    w = anchors[a, 0] * np.exp(raw[2]) / grid_size
                    h = anchors[a, 1] * np.exp(raw[3]) / grid_size
                    cls_scores = _sigmoid(raw[5:])
                    label = np.argmax(cls_scores)
                    score = obj * cls_scores[label]
                    boxes.append([cx, cy, w, h])
                    scores.append(score)
                    labels.append(label)

        if boxes:
            boxes = np.array(boxes)
            scores = np.array(scores)
            labels = np.array(labels)
            # 按类别做 NMS
            keep = []
            for c in range(num_classes):
                cls_mask = labels == c
                if not np.any(cls_mask):
                    continue
                cls_idx = np.where(cls_mask)[0]
                cls_keep = nms(boxes[cls_idx], scores[cls_idx], iou_threshold=0.5)
                keep.extend(cls_idx[cls_keep].tolist())
            if keep:
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
        else:
            boxes = np.zeros((0, 4))
            scores = np.zeros(0)
            labels = np.zeros(0, dtype=np.int64)

        results.append({'boxes': boxes, 'scores': scores, 'labels': labels})
    return results


# ============================================================
# YOLOv3-lite 模型
# ============================================================
@register_model('yolov3_lite')
class YOLOv3Lite(Module):
    """YOLOv3 轻量版（单尺度目标检测）。

    Args:
        in_channels: 输入通道数（默认 1，灰度图）
        num_classes: 类别数（默认 3，适配 ShapesDetection）
        anchors: anchor 宽高列表（特征图像素单位）
        grid_size: 特征图边长（默认 8，对应 32×32 输入经过 2 次下采样）
    """

    def __init__(self, in_channels=1, num_classes=3, anchors=None, grid_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.anchors = np.asarray(anchors if anchors is not None else DEFAULT_ANCHORS,
                                   dtype=np.float64)
        self.num_anchors = len(self.anchors)
        self.grid_size = grid_size
        self.bbox_attrs = 5 + num_classes

        # 轻量 backbone：4 个卷积块
        self.backbone = Sequential(
            Conv2d(in_channels, 16, 3, stride=1, padding=1),
            BatchNorm2d(16),
            LeakyReLU(0.1),
            Conv2d(16, 32, 3, stride=2, padding=1),   # 32→16
            BatchNorm2d(32),
            LeakyReLU(0.1),
            Conv2d(32, 64, 3, stride=2, padding=1),   # 16→8
            BatchNorm2d(64),
            LeakyReLU(0.1),
            Conv2d(64, 64, 3, stride=1, padding=1),   # 8→8
            BatchNorm2d(64),
            LeakyReLU(0.1),
        )

        # 检测头：1×1 卷积输出每个 anchor 的 (5 + num_classes)
        self.head = Conv2d(64, self.num_anchors * self.bbox_attrs,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """前向传播。

        Args:
            x: (batch, in_channels, H, W)，默认 (batch, 1, 32, 32)

        Returns:
            predictions: (batch, num_anchors*(5+num_classes), grid_size, grid_size)
        """
        feat = self.backbone(x)
        return self.head(feat)

    def make_loss(self):
        """创建对应的 YOLOLoss 实例。"""
        return YOLOLoss(self.anchors, self.grid_size, self.num_classes)

    def decode(self, predictions_np, conf_threshold=0.5, img_size=32):
        """推理解码（输入 numpy，输出边界框列表）。"""
        return decode_predictions(predictions_np, self.anchors, self.grid_size,
                                  self.num_classes, conf_threshold, img_size)

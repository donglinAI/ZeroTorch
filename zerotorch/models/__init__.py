"""模型库：按任务组织，全部注册到 MODEL_REGISTRY（即插即用）。

教学演示版：只保留 3 个核心模型
- ResNetLite: 图像分类
- YOLOv3Lite: 目标检测
- ViTLite: 视觉 Transformer 分类
"""
from .registry import MODEL_REGISTRY, register_model
from .resnet import ResNetLite
from .yolo import YOLOv3Lite
from .vit import ViTLite

__all__ = ['MODEL_REGISTRY', 'register_model', 'ResNetLite',
           'YOLOv3Lite', 'ViTLite']

"""模型库：按任务组织，全部注册到 MODEL_REGISTRY（即插即用）。

v1 旧模型（MLP/LeNet5/ShapeDetector/CharLSTM/VAE）已从导出移除，文件保留待清理。
"""
from .registry import MODEL_REGISTRY, register_model
from .vit import ViTLite
from .resnet import ResNetLite
from .yolo import YOLOv3Lite
from .bert import BERTLite
from .unet import UNetLite
from .gpt import GPTLite
from .dcgan import DCGANLite
from .multimodal import MultimodalLite

__all__ = ['MODEL_REGISTRY', 'register_model', 'ViTLite', 'ResNetLite',
           'YOLOv3Lite', 'BERTLite', 'UNetLite', 'GPTLite',
           'DCGANLite', 'MultimodalLite']

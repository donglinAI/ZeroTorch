"""内置数据集。"""
from .mnist import MNIST
from .synthetic import (make_blobs, make_moons, ShapesDetection,
                        CharTextDataset, load_sample_text)

__all__ = [
    'MNIST', 'make_blobs', 'make_moons', 'ShapesDetection',
    'CharTextDataset', 'load_sample_text',
]

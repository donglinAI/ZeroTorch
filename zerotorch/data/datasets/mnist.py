"""MNIST 数据集：自动下载（多镜像回退）、解析 idx 格式、本地 npz 缓存。"""
from __future__ import annotations

import gzip
import os
import urllib.request

import numpy as np

from ..dataset import Dataset

_MIRRORS = [
    'https://raw.githubusercontent.com/fgnt/mnist/master/{name}',
    'https://ossci-datasets.s3.amazonaws.com/mnist/{name}',
    'https://github.com/myleott/mnist_png/raw/master/{name}',
]

_FILES = {
    'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    'test': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'),
}


def _download(url, dest):
    print(f'  下载 {url} -> {dest}')
    urllib.request.urlretrieve(url, dest)


def _read_idx(gz_path, is_image):
    with gzip.open(gz_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n = int.from_bytes(f.read(4), 'big')
        if is_image:
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        else:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


class MNIST(Dataset):
    """MNIST 手写数字数据集。

    首次使用自动下载并缓存为 .npz；所有镜像失败时抛出异常
    （示例代码会回退到合成数据）。
    """

    def __init__(self, root='./data', train=True, download=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        split = 'train' if train else 'test'
        cache = os.path.join(root, f'mnist_{split}.npz')
        if os.path.exists(cache):
            data = np.load(cache)
            self.images, self.labels = data['images'], data['labels']
            return
        img_name, lbl_name = _FILES[split]
        img_path = os.path.join(root, img_name)
        lbl_path = os.path.join(root, lbl_name)
        if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
            if not download:
                raise FileNotFoundError(f'MNIST 文件不存在：{root}')
            os.makedirs(root, exist_ok=True)
            ok = False
            for mirror in _MIRRORS:
                try:
                    _download(mirror.format(name=img_name), img_path)
                    _download(mirror.format(name=lbl_name), lbl_path)
                    ok = True
                    break
                except Exception as e:      # noqa: BLE001
                    print(f'  镜像失败：{mirror} ({e})')
                    for p in (img_path, lbl_path):
                        if os.path.exists(p):
                            os.remove(p)
            if not ok:
                raise RuntimeError('MNIST 所有下载源均失败，请检查网络或使用合成数据')
        images = _read_idx(img_path, True).astype(np.float32) / 255.0
        labels = _read_idx(lbl_path, False).astype(np.int64)
        images = images[:, None, :, :]      # (N, 1, 28, 28)
        os.makedirs(root, exist_ok=True)
        np.savez(cache, images=images, labels=labels)
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

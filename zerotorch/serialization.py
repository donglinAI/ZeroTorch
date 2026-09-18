"""序列化：模型 / 优化器 / checkpoint 的保存与加载（pickle 封装）。"""
from __future__ import annotations

import os
import pickle


def save(obj, path):
    """保存任意对象（对标 torch.save）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    """加载对象（对标 torch.load）。"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(path, model, optimizer=None, epoch=None, history=None,
                    extra=None):
    """保存 checkpoint：模型 state_dict + 优化器状态 + 元信息。"""
    ckpt = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'history': history,
        'extra': extra or {},
    }
    if optimizer is not None:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
    save(ckpt, path)
    return path


def load_checkpoint(path, model=None, optimizer=None, map_location=None):
    """加载 checkpoint 并回填模型 / 优化器状态，返回完整 dict。"""
    ckpt = load(path)
    if model is not None and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt

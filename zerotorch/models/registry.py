"""模型注册机制。"""
from __future__ import annotations

MODEL_REGISTRY = {}


def register_model(name):
    """注册模型类，支持按名字实例化（即插即用）。"""
    def deco(cls):
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return deco

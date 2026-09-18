"""量化压缩模块：INT8 权重量化。

教学版：对称量化（FP32 → INT8），权重量化（不激活量化），
量化后推理验证精度损失，对比模型大小。
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor
from .nn.module import Module


def quantize_tensor_fp32_to_int8(w: np.ndarray):
    """对称量化 FP32 → INT8。"""
    max_val = np.max(np.abs(w))
    if max_val == 0:
        return np.zeros_like(w, dtype=np.int8), 1.0, 0
    scale = max_val / 127.0
    w_int8 = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return w_int8, scale, 0


def dequantize_tensor_int8_to_fp32(w_int8: np.ndarray, scale: float, zero_point: int = 0):
    """反量化 INT8 → FP32。"""
    return (w_int8.astype(np.float32) - zero_point) * scale


def quantize_model(model: Module):
    """量化整个模型的权重（INT8 对称量化）。"""
    quantized_params = {}
    total_original = 0
    total_quantized = 0
    for name, param in model.named_parameters():
        w = param.data.astype(np.float32)
        w_int8, scale, zp = quantize_tensor_fp32_to_int8(w)
        quantized_params[name] = (w_int8, scale, zp)
        total_original += w.size * 4
        total_quantized += w_int8.size * 1 + 4 + 1
    original_size_mb = round(total_original / 1024 / 1024, 3)
    quantized_size_mb = round(total_quantized / 1024 / 1024, 3)
    compression_ratio = round(original_size_mb / quantized_size_mb, 2) if quantized_size_mb > 0 else 0
    return quantized_params, original_size_mb, quantized_size_mb, compression_ratio


def dequantize_model(model: Module, quantized_params: dict):
    """反量化（把 INT8 权重复原成 FP32，写入模型）。"""
    for name, param in model.named_parameters():
        if name in quantized_params:
            w_int8, scale, zp = quantized_params[name]
            w_fp32 = dequantize_tensor_int8_to_fp32(w_int8, scale, zp)
            param.data = w_fp32.astype(np.float64)


def compute_quantization_error(model: Module, quantized_params: dict) -> dict:
    """计算量化误差（平均/最大相对误差）。"""
    errors = []
    for name, param in model.named_parameters():
        if name in quantized_params:
            w_original = param.data.astype(np.float32)
            w_int8, scale, zp = quantized_params[name]
            w_dequant = dequantize_tensor_int8_to_fp32(w_int8, scale, zp)
            rel_error = np.mean(np.abs(w_original - w_dequant) / (np.abs(w_original) + 1e-8))
            errors.append(rel_error)
    return {
        'avg_rel_error': round(float(np.mean(errors)), 6),
        'max_rel_error': round(float(np.max(errors)), 6),
    }
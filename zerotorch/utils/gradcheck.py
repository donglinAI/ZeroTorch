"""数值梯度检查工具：用中心差分验证解析梯度，是引擎正确性的基石。"""
from __future__ import annotations

import numpy as np

from ..tensor import Tensor


def numerical_gradient(fn, inputs, eps=1e-6):
    """对每个输入张量计算数值梯度（中心差分）。

    fn: 接收若干 Tensor、返回标量 Tensor 的函数。
    返回: 与 inputs 等长的 numpy 数组列表。
    """
    grads = []
    for inp in inputs:
        data = inp.data.astype(np.float64).copy()
        g = np.zeros_like(data)
        flat = data.reshape(-1)
        it = np.nditer(flat, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old = flat[idx]
            flat[idx] = old + eps
            plus = _to_scalar(fn, inputs, inp, data)
            flat[idx] = old - eps
            minus = _to_scalar(fn, inputs, inp, data)
            flat[idx] = old
            g.reshape(-1)[idx] = (plus - minus) / (2 * eps)
            it.iternext()
        grads.append(g)
    return grads


def _to_scalar(fn, inputs, target, new_data):
    """用 new_data 替换 target 的数据后执行 fn，返回标量（float64 全程计算）。"""
    old = target.data
    target.data = new_data          # 保持 float64，避免 float32 舍入吃掉扰动
    try:
        out = fn(*inputs)
        val = float(out.data)
    finally:
        target.data = old
    return val


def gradcheck(fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-3, verbose=False):
    """校验 fn 的解析梯度与数值梯度一致。

    返回 (通过?, 最大误差, 误差明细列表)。
    """
    analytic = []
    for inp in inputs:
        inp.zero_grad()
    loss = fn(*inputs)
    loss.backward()
    for inp in inputs:
        analytic.append(inp.grad if inp.grad is not None else np.zeros_like(inp.data))

    numeric = numerical_gradient(fn, inputs, eps=eps)
    details = []
    max_err = 0.0
    ok = True
    for i, (a, n) in enumerate(zip(analytic, numeric)):
        a = np.asarray(a, dtype=np.float64)
        denom = np.maximum(np.abs(n), 1.0)
        err = np.abs(a - n) / denom
        m = float(err.max()) if err.size else 0.0
        max_err = max(max_err, m)
        details.append((i, m, float(np.abs(a - n).max())))
        if m > atol + rtol:
            ok = False
        if verbose:
            print(f'  input[{i}]: 相对误差={m:.3e}  绝对误差={details[-1][2]:.3e}')
    return ok, max_err, details

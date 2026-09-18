"""核心：Tensor 数据结构 + 自动微分引擎（动态图 autograd，风格对齐 PyTorch）。

设计要点
--------
1. Tensor 是计算图中的节点，持有：
   - ``data``          : numpy 数组（float32）
   - ``grad``          : 反向传播累积的梯度（numpy 数组），默认 None
   - ``requires_grad`` : 是否需要梯度
   - ``_grad_fn``      : 反向节点 ``(FunctionClass, ctx, inputs)``，叶子为 None

2. 算子体系：每个自定义算子继承 :class:`Function`，只需实现
   ``forward(ctx, *raw_args) -> np.ndarray`` 和
   ``backward(ctx, grad_output) -> tuple`` 两个静态方法，
   即可通过 ``Xxx.apply(*tensors_or_scalars)`` 自动获得"前向建图 + 反向求导"能力。
   新增一个算子 = 新增一个 Function 子类，即插即用。

3. 反向传播：``Tensor.backward()`` 采用"后序 DFS 拓扑排序 + 逆拓扑传播"，
   支持 DAG（同一张量被多个算子复用，梯度累加）与 numpy 广播（自动规约维度）。
"""
from __future__ import annotations

import numpy as np


def _any_requires_grad(x):
    """递归判断参数（含 Tensor 列表）中是否有需要梯度的 Tensor。"""
    if isinstance(x, Tensor):
        return x.requires_grad
    if isinstance(x, (list, tuple)):
        return any(_any_requires_grad(y) for y in x)
    return False


# ---------------------------------------------------------------------------
# 自动微分的"上下文"与算子基类
# ---------------------------------------------------------------------------
class Context:
    """前向计算与反向计算之间的便签本：前向时把 backward 需要的中间量存进来。"""

    def __init__(self):
        self._saved = {}

    def save_for_backward(self, **kwargs):
        self._saved.update(kwargs)

    def get(self, key, default=None):
        return self._saved.get(key, default)

    def __getitem__(self, key):
        return self._saved[key]


class Function:
    """自定义算子基类。

    子类只需实现两个静态方法：
        forward(ctx, *raw_args) -> np.ndarray
        backward(ctx, grad_output) -> tuple  # 与 args 一一对齐，非 Tensor 参数返回 None
    然后调用 ``ClassName.apply(*args)`` 即可，apply 负责：
    1) 把 Tensor 参数解包成 numpy 数组传给 forward；
    2) 依据输入是否 requires_grad 决定输出是否挂反向节点；
    3) 构造输出 Tensor 并保存 (FunctionClass, ctx, args) 以便 backward 重建梯度。
    """

    @classmethod
    def apply(cls, *args):
        ctx = Context()
        raw = []
        for a in args:
            if isinstance(a, Tensor):
                raw.append(a.data)
            elif isinstance(a, (list, tuple)):
                raw.append([x.data if isinstance(x, Tensor) else x for x in a])
            else:
                raw.append(a)
        out_data = cls.forward(ctx, *raw)
        requires_grad = any(_any_requires_grad(a) for a in args)
        out = Tensor(np.asarray(out_data), requires_grad=requires_grad)
        if requires_grad:
            out._grad_fn = (cls, ctx, args)
        return out


def _sum_to_shape(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """把梯度规约到目标形状，处理 numpy 广播产生的额外维度。

    规则：先对多出来的前导维度求和，再对 shape 中为 1 的维度求和。
    """
    if grad.shape == tuple(shape):
        return grad
    if len(shape) == 0:                      # 目标为标量
        return np.asarray(grad.sum())
    while grad.ndim > len(shape):            # 去掉广播补的前导维
        grad = grad.sum(axis=0)
    for axis, s in enumerate(shape):         # 对 size==1 的维度求和
        if s == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------
class Tensor:
    """n 维数组 + 自动微分。API 风格对齐 PyTorch。

    用法示例::

        x = nf.tensor(np.random.randn(4, 4), requires_grad=True)
        y = (x * x).sum()
        y.backward()
        print(x.grad)   # == 2 * x
    """

    __array_priority__ = 1000          # 让 np.ndarray op Tensor 时优先走 Tensor
    _global_dtype = np.float32

    def __init__(self, data, requires_grad=False, dtype=None, device='cpu',
                 _grad_fn=None, _is_leaf=None):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, np.ndarray):
            # 已有数组保留原 dtype（对齐 PyTorch），仅显式指定时才转换
            arr = data if dtype is None else data.astype(dtype)
        else:
            arr = np.asarray(data, dtype=dtype if dtype is not None else self._global_dtype)
        self.data = arr
        self.grad = None                       # 累积梯度
        self.requires_grad = bool(requires_grad)
        self.device = device
        self._grad_fn = _grad_fn               # (FunctionClass, ctx, inputs) 或 None
        self._is_leaf = _is_leaf if _is_leaf is not None else (_grad_fn is None)

    # ------------------------- 基础属性 -------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return transpose(self)

    @property
    def grad_fn(self):
        return self._grad_fn[0].__name__ if self._grad_fn is not None else None

    @property
    def is_leaf(self):
        return self._is_leaf

    def size(self, dim=None):
        return self.data.size if dim is None else self.data.shape[dim]

    def numpy(self):
        """返回底层 numpy 数组（视图，谨慎修改）。"""
        return self.data

    def __array__(self, dtype=None):
        """支持 np.asarray(tensor) / np.array(tensor)。"""
        return np.asarray(self.data, dtype=dtype) if dtype is not None else self.data

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def detach(self):
        """返回不追踪梯度的新 Tensor（共享数据）。"""
        return Tensor(self.data, requires_grad=False)

    def requires_grad_(self, flag=True):
        self.requires_grad = bool(flag)
        return self

    def zero_grad(self):
        self.grad = None
        return self

    # ------------------------- 反向传播 -------------------------
    def backward(self, gradient=None):
        """从当前 Tensor 出发沿计算图反向传播梯度。

        gradient: 上游梯度；标量可省略。叶子 / 非叶子张量的 ``.grad``
        都会被填上累积梯度（便于调试与可视化）。
        """
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    'backward() 需要 gradient 参数：当前 Tensor 不是标量')
            gradient = np.ones_like(self.data)

        grads = {id(self): gradient}
        topo, visited = [], set()

        # 后序 DFS：保证子节点（更靠近输入的算子）排在父节点之前
        def build(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            if t._grad_fn is not None:
                _, _, inputs = t._grad_fn
                for inp in inputs:
                    if isinstance(inp, Tensor):
                        build(inp)
                    elif isinstance(inp, (list, tuple)):
                        for x in inp:
                            if isinstance(x, Tensor):
                                build(x)
            topo.append(t)

        build(self)

        # 逆拓扑序传播：先算最靠近 loss 的节点
        def _accumulate(t, tg):
            if not isinstance(t, Tensor) or not t.requires_grad or tg is None:
                return
            if id(t) in grads:
                grads[id(t)] = grads[id(t)] + tg
            else:
                grads[id(t)] = tg

        for t in reversed(topo):
            if t._grad_fn is None:
                continue
            fn_cls, ctx, inputs = t._grad_fn
            g = grads.get(id(t))
            if g is None:
                continue
            out_grads = fn_cls.backward(ctx, g)
            if not isinstance(out_grads, (tuple, list)):
                out_grads = (out_grads,)
            for inp, ig in zip(inputs, out_grads):
                if isinstance(inp, (list, tuple)) and isinstance(ig, (list, tuple)):
                    for t, tg in zip(inp, ig):
                        _accumulate(t, tg)
                else:
                    _accumulate(inp, ig)

        # 回填梯度
        for t in topo:
            if t.requires_grad:
                t.grad = grads.get(id(t))

    # ------------------------- 算子转发（运算符重载） -------------------------
    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return Add.apply(other, self)

    def __sub__(self, other):
        return Sub.apply(self, other)

    def __rsub__(self, other):
        return Sub.apply(other, self)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(other, self)

    def __truediv__(self, other):
        return Div.apply(self, other)

    def __rtruediv__(self, other):
        return Div.apply(other, self)

    def __neg__(self):
        return Neg.apply(self)

    def __pow__(self, exponent):
        return Pow.apply(self, exponent)

    def __matmul__(self, other):
        return Matmul.apply(self, other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __getitem__(self, item):
        raise NotImplementedError(
            'Tensor 暂不支持直接下标索引，请使用 F.narrow / F.gather / F.reshape 等函数。')

    def __bool__(self):
        raise TypeError('Tensor 的布尔值有歧义，请用 .item() 或 .data 判断。')

    # ------------------------- 常用函数式方法 -------------------------
    def sum(self, dim=None, keepdims=False):
        from .functions import sum as _sum
        return _sum(self, dim, keepdims)

    def mean(self, dim=None, keepdims=False):
        from .functions import mean as _mean
        return _mean(self, dim, keepdims)

    def max(self, dim=None, keepdims=False):
        from .functions import max as _max
        return _max(self, dim, keepdims)

    def reshape(self, shape):
        from .functions import reshape as _reshape
        return _reshape(self, shape)

    def flatten(self, start_dim=1, end_dim=-1):
        from .functions import flatten as _flatten
        return _flatten(self, start_dim, end_dim)

    def transpose(self, axes=None):
        from .functions import transpose as _transpose
        return _transpose(self, axes)

    def view(self, shape):
        return self.reshape(shape)

    def exp(self):
        from .functions import exp as _exp
        return _exp(self)

    def log(self):
        from .functions import log as _log
        return _log(self)

    def sqrt(self):
        from .functions import sqrt as _sqrt
        return _sqrt(self)

    def matmul(self, other):
        return Matmul.apply(self, other)

    def relu(self):
        from .functions import relu as _relu
        return _relu(self)

    def softmax(self, dim=-1):
        from .functions import softmax as _softmax
        return _softmax(self, dim)

    # ------------------------- 展示 -------------------------
    def __repr__(self):
        parts = []
        if self._grad_fn is not None:
            parts.append(f'grad_fn=<{self.grad_fn}>')
        if self.requires_grad:
            parts.append('requires_grad=True')
        body = np.array2string(self.data, precision=4, threshold=12, separator=', ')
        return f'Tensor({body}, {", ".join(parts)})'

    __str__ = __repr__

    # ------------------------- 全局配置 -------------------------
    @classmethod
    def set_default_dtype(cls, dtype):
        cls._global_dtype = np.dtype(dtype)


# # 运算符重载所需的算子（在底部导入以规避循环依赖）
from .ops.basic import Add, Sub, Mul, Div, Pow, Neg          # noqa: E402
from .ops.matrix import Matmul                               # noqa: E402
from .ops.shape import transpose                             # noqa: E402


# 便捷构造函数
def tensor(data, requires_grad=False, dtype=None, device='cpu'):
    """构造 Tensor。"""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)

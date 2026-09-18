"""nn 模块基类：Module / Parameter，含参数注册、state_dict、train/eval 等。"""
from __future__ import annotations

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """可学习参数：默认 requires_grad=True，且是叶子张量。"""

    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad, _is_leaf=True)


class Module:
    """神经网络模块基类（对标 torch.nn.Module）。

    - 把 ``Parameter`` / ``Module`` 属性自动注册到 ``_parameters`` / ``_modules``；
    - 提供 ``parameters()``、``state_dict()``、``load_state_dict()``、``train()/eval()`` 等；
    - 子类实现 ``forward()``，通过 ``__call__`` 调用。
    """

    def __init__(self):
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, 'training', True)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
            object.__setattr__(self, name, value)      # 同时挂为实例属性，便于 self.weight 访问
        elif isinstance(value, Module):
            self._modules[name] = value
            object.__setattr__(self, name, value)
        elif name in ('_parameters', '_modules', '_buffers', 'training'):
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    # ------------------------- 注册 -------------------------
    def register_parameter(self, name, param):
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_buffer(self, name, arr):
        """注册不参与梯度更新的 buffer（如 BN 的 running_mean/var）。"""
        arr = np.asarray(arr)
        self._buffers[name] = arr
        object.__setattr__(self, name, arr)

    def register_module(self, name, module):
        self._modules[name] = module
        object.__setattr__(self, name, module)

    # ------------------------- 遍历 -------------------------
    def named_parameters(self, prefix=''):
        for name, p in self._parameters.items():
            yield f'{prefix}.{name}' if prefix else name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(f'{prefix}.{name}' if prefix else name)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_buffers(self, prefix=''):
        for name, b in self._buffers.items():
            yield f'{prefix}.{name}' if prefix else name, b
        for name, m in self._modules.items():
            yield from m.named_buffers(f'{prefix}.{name}' if prefix else name)

    def buffers(self):
        for _, b in self.named_buffers():
            yield b

    def named_modules(self, prefix=''):
        yield prefix or '', self
        for name, m in self._modules.items():
            yield from m.named_modules(f'{prefix}.{name}' if prefix else name)

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def children(self):
        for m in self._modules.values():
            yield m

    # ------------------------- 状态管理 -------------------------
    def state_dict(self):
        sd = {}
        for name, p in self.named_parameters():
            sd[name] = p.data
        for name, b in self.named_buffers():
            sd[name] = b
        return sd

    def _param_and_buffer_map(self):
        mapping = {}
        for name, p in self.named_parameters():
            mapping[name] = p
        for name, b in self.named_buffers():
            mapping[name] = b
        return mapping

    def load_state_dict(self, state_dict, strict=True):
        own = set(self.state_dict().keys())
        keys = set(state_dict.keys())
        if strict:
            missing = own - keys
            unexpected = keys - own
            if missing or unexpected:
                raise RuntimeError(
                    f'load_state_dict 不匹配：missing={sorted(missing)} '
                    f'unexpected={sorted(unexpected)}')
        mapping = self._param_and_buffer_map()
        for name, val in state_dict.items():
            if name not in mapping:
                continue
            target = mapping[name]
            arr = np.asarray(val)
            if isinstance(target, Parameter):
                target.data = arr.astype(target.data.dtype)
            else:
                np.copyto(target, arr.astype(target.dtype))

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None
        return self

    def apply(self, fn):
        fn(self)
        for m in self.modules():
            if m is not self:
                m.apply(fn)
        return self

    def to(self, device='cpu'):
        for m in self.modules():
            m.device = device
        return self

    def num_parameters(self, trainable_only=False):
        total = 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            total += p.data.size
        return total

    # ------------------------- 前向 -------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        lines = [self.__class__.__name__ + '(']
        for name, m in self._modules.items():
            lines.append(f'  ({name}): {m.__class__.__name__}')
        for name, p in self._parameters.items():
            lines.append(f'  ({name}): Parameter{tuple(p.shape)}')
        lines.append(')')
        return '\n'.join(lines)

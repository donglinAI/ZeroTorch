"""容器模块：Sequential / ModuleList。"""
from __future__ import annotations

from collections import OrderedDict

from .module import Module


class Sequential(Module):
    """按顺序执行子模块（对标 torch.nn.Sequential）。

    用法::

        model = Sequential(
            Linear(784, 128), ReLU(),
            Linear(128, 10),
        )
    """

    def __init__(self, *modules):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (dict, OrderedDict)):
            for name, m in modules[0].items():
                self.add_module(name, m)
        else:
            for i, m in enumerate(modules):
                self.add_module(str(i), m)

    def add_module(self, name, module):
        self.register_module(name, module)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class ModuleList(Module):
    """模块列表（对标 torch.nn.ModuleList）。"""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                self.register_module(str(i), m)

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)

    def append(self, module):
        self.register_module(str(len(self._modules)), module)
        return self

    def forward(self, x):
        raise NotImplementedError('ModuleList 仅作容器，请自行遍历调用。')

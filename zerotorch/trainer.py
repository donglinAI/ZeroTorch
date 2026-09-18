"""训练器：统一训练/验证循环，基于 Callback 钩子扩展（对标 torch 生态的 Trainer 理念）。"""
from __future__ import annotations

import time

import numpy as np

from .metrics import METRIC_REGISTRY


class Callback:
    """训练过程回调基类：在关键节点被 Trainer 调用，便于扩展日志/存档/可视化。"""

    def on_train_begin(self, trainer): ...
    def on_train_end(self, trainer): ...
    def on_epoch_begin(self, trainer, epoch): ...
    def on_epoch_end(self, trainer, epoch, logs): ...
    def on_batch_end(self, trainer, batch, logs): ...


class History(Callback):
    """记录每个 epoch 的 loss / 指标，供可视化使用。"""

    def __init__(self):
        self.history = {'epoch': [], 'loss': [], 'val_loss': []}

    def on_train_begin(self, trainer):
        if trainer.metrics:
            for name in trainer.metrics:
                self.history.setdefault(name, [])
                self.history.setdefault(f'val_{name}', [])

    def on_epoch_end(self, trainer, epoch, logs):
        for key, val in logs.items():
            self.history.setdefault(key, []).append(val)


class ModelCheckpoint(Callback):
    """按监控指标保存最优/最新 checkpoint。"""

    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = np.inf if mode == 'min' else -np.inf

    def on_epoch_end(self, trainer, epoch, logs):
        if self.monitor not in logs:
            return
        value = logs[self.monitor]
        better = value < self.best if self.mode == 'min' else value > self.best
        if better:
            self.best = value
            from .serialization import save_checkpoint
            save_checkpoint(self.filepath, trainer.model, trainer.optimizer,
                            epoch=epoch, history=trainer.history.history if trainer.history else None)
            print(f'  [checkpoint] 保存最优模型 -> {self.filepath} ({self.monitor}={value:.4f})')


class ProgressBar(Callback):
    """每个 epoch 结束时打印一行摘要。"""

    def on_epoch_end(self, trainer, epoch, logs):
        parts = [f'epoch {epoch + 1}/{trainer.epochs}']
        parts.append(f"loss={logs.get('loss', float('nan')):.4f}")
        if trainer.metrics:
            for name in trainer.metrics:
                parts.append(f"{name}={logs.get(name, float('nan')):.4f}")
        if 'val_loss' in logs:
            parts.append(f"val_loss={logs['val_loss']:.4f}")
            if trainer.metrics:
                for name in trainer.metrics:
                    parts.append(f"val_{name}={logs.get('val_' + name, float('nan')):.4f}")
        print('  ' + ' | '.join(parts))


class Trainer:
    """训练器。

    用法::

        trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader,
                          epochs=10, metrics=['accuracy'], callbacks=[...])
        history = trainer.fit()

    支持 DataParallel 包装的模型（内部自动切换为分片求梯度路径）。
    """

    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader=None,
                 epochs=1, metrics=None, callbacks=None, grad_clip=None,
                 scheduler=None, device='cpu'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.device = device
        self.history = None
        self.current_epoch = 0

    # ------------------------- 回调调度 -------------------------
    def _call(self, method, *args):
        for cb in self.callbacks:
            getattr(cb, method)(self, *args)

    # ------------------------- 主流程 -------------------------
    def fit(self):
        histories = [cb for cb in self.callbacks if isinstance(cb, History)]
        if histories:
            self.history = histories[0]
        else:
            history = History()
            self.history = history
            self.callbacks = [history] + self.callbacks
        self._call('on_train_begin')
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self._call('on_epoch_begin', epoch)
            train_logs = self._run_epoch(self.train_loader, training=True)
            logs = dict(train_logs)
            if self.val_loader is not None:
                val_logs = self.evaluate(self.val_loader)
                logs.update({'val_' + k: v for k, v in val_logs.items()})
            if self.scheduler is not None:
                self.scheduler.step()
            self._call('on_epoch_end', epoch, logs)
        self._call('on_train_end')
        return self.history.history

    def _run_epoch(self, loader, training=True):
        total_loss, n = 0.0, 0
        metric_sums = {name: 0.0 for name in self.metrics}
        if training:
            self.model.train()
        else:
            self.model.eval()
        start = time.time()
        for i, (x, y) in enumerate(loader):
            if training:
                self.optimizer.zero_grad()
                loss = self._train_step(x, y)
                if self.grad_clip is not None:
                    from .optim.base import clip_grad_norm_
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            else:
                # 验证态不需要反向
                if getattr(self.model, 'is_data_parallel', False):
                    loss = self.model(x)
                    loss = self.loss_fn(loss, y)
                else:
                    loss = self.loss_fn(self.model(x), y)
            loss_val = float(loss.data)
            total_loss += loss_val
            n += 1
            if self.metrics:
                pred = self._predict(x)
                for name in self.metrics:
                    metric_sums[name] += METRIC_REGISTRY[name](pred, y.data if hasattr(y, 'data') else y)
            self._call('on_batch_end', i, {'loss': loss_val})
        logs = {'loss': total_loss / max(n, 1), 'time': time.time() - start}
        for name in self.metrics:
            logs[name] = metric_sums[name] / max(n, 1)
        return logs

    def _train_step(self, x, y):
        """单步：DataParallel 模型走分片梯度路径，普通模型走标准前反向。"""
        if getattr(self.model, 'is_data_parallel', False):
            loss = self.model.train_step(x, y, self.loss_fn)
            return loss
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        return loss

    def _predict(self, x):
        pred = self.model(x)
        return pred.data if hasattr(pred, 'data') else pred

    def evaluate(self, loader):
        return self._run_epoch(loader, training=False)

    def predict(self, loader):
        """推理：返回 (inputs, predictions) 的 numpy 数组列表。"""
        self.model.eval()
        xs, preds = [], []
        for x, _ in loader:
            xs.append(x.data if hasattr(x, 'data') else x)
            preds.append(self._predict(x))
        return xs, preds

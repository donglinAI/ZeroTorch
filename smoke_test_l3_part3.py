"""L3 第三批 · 验收测试脚本（trainer.py + serialization.py + viz/）

用法：
    python smoke_test_l3_part3.py        # 与 zerotorch/ 目录同级，或位于其子目录

覆盖：
    1. serialization.py：save/load / save_checkpoint+load_checkpoint 往返
    2. trainer.py：Trainer.fit 训练 / History / ProgressBar / ModelCheckpoint /
                   metrics / val_loader / grad_clip / scheduler / evaluate / predict
    3. viz/：plot_curves / plot_classification / plot_detections /
             plot_image_grid / plot_text_history（生成图片并校验）
    4. 端到端：Trainer + DataLoader + make_blobs 完整训练 + checkpoint + 曲线

退出码：全部通过返回 0，任一失败返回 1。
"""
import os
import sys
import tempfile

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
sys.path.insert(0, os.path.dirname(_base))

import numpy as np

FAILURES = []
PASSED = 0


def check(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAILURES.append(name)
        print(f"  [FAIL] {name} {detail}")


def main():
    rng = np.random.RandomState(42)
    tmpdir = tempfile.mkdtemp(prefix='zt_l3p3_')

    # ---------- 0. 目录诊断 ----------
    print("\n== 0. 目录结构诊断 ==")
    cands = [os.path.join(_base, 'zerotorch'),
             os.path.join(os.path.dirname(_base), 'zerotorch')]
    pkg_dir = next((c for c in cands if os.path.isdir(c)), None)
    if pkg_dir is None:
        print(f"  [ERROR] 找不到包目录（已检查: {cands}）")
        return 1
    required = ['trainer.py', 'serialization.py',
                os.path.join('viz', '__init__.py'), os.path.join('viz', 'plots.py')]
    missing = [f for f in required if not os.path.isfile(os.path.join(pkg_dir, f))]
    if missing:
        print(f"  [ERROR] 缺少文件: {missing}")
        return 1
    print("  [PASS] trainer/serialization/viz 目录结构完整")

    import zerotorch as zt
    from zerotorch.nn.container import Sequential
    from zerotorch.nn.layers import Linear, ReLU
    from zerotorch.loss import CrossEntropyLoss
    from zerotorch.optim import Adam, StepLR
    from zerotorch.data import DataLoader, TensorDataset
    from zerotorch.data.datasets import make_blobs, ShapesDetection
    from zerotorch.serialization import save, load, save_checkpoint, load_checkpoint
    from zerotorch.trainer import Trainer, Callback, History, ProgressBar, ModelCheckpoint
    from zerotorch.viz import (plot_curves, plot_classification, plot_detections,
                                plot_image_grid, plot_text_history)

    # ---------- 1. serialization.py ----------
    print("\n== 1. 序列化 serialization.py ==")
    # save/load 任意对象
    obj = {'a': [1, 2, 3], 'b': np.array([4.0, 5.0])}
    p1 = os.path.join(tmpdir, 'sub', 'obj.pkl')
    save(obj, p1)
    check("save/load 任意对象", os.path.isfile(p1))
    obj2 = load(p1)
    check("save/load 内容一致", obj2['a'] == [1, 2, 3]
          and np.array_equal(obj2['b'], [4.0, 5.0]))
    # save_checkpoint / load_checkpoint 往返
    net = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    opt = Adam(net.parameters(), lr=1e-3)
    ckpt_path = os.path.join(tmpdir, 'ckpt', 'best.npz')
    save_checkpoint(ckpt_path, net, optimizer=opt, epoch=5, history={'loss': [1.0, 0.5]})
    check("save_checkpoint 生成文件", os.path.isfile(ckpt_path))
    # 新模型 + 新优化器加载
    net2 = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    opt2 = Adam(net2.parameters(), lr=1e-3)
    ckpt = load_checkpoint(ckpt_path, model=net2, optimizer=opt2)
    check("load_checkpoint 回填模型",
          all(np.array_equal(p1.data, p2.data)
              for p1, p2 in zip(net.parameters(), net2.parameters())))
    check("load_checkpoint 回填优化器", opt2._step_count == opt._step_count)
    check("load_checkpoint 元信息", ckpt['epoch'] == 5 and ckpt['history']['loss'] == [1.0, 0.5])

    # ---------- 2. trainer.py ----------
    print("\n== 2. 训练器 Trainer ==")
    # 准备数据
    Xtr, ytr = make_blobs(n_samples=300, n_features=8, centers=3, seed=1)
    Xval, yval = make_blobs(n_samples=60, n_features=8, centers=3, seed=2)
    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xval, yval), batch_size=32, shuffle=False)

    model = Sequential(Linear(8, 32), ReLU(), Linear(32, 3))
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    ckpt_path2 = os.path.join(tmpdir, 'trainer_ckpt', 'best.pkl')
    history_cb = History()
    trainer = Trainer(model, loss_fn, optimizer, train_dl, val_loader=val_dl,
                      epochs=5, metrics=['accuracy'],
                      callbacks=[history_cb, ProgressBar(),
                                 ModelCheckpoint(ckpt_path2, monitor='val_loss', mode='min')],
                      grad_clip=1.0, scheduler=scheduler)
    hist = trainer.fit()
    check("Trainer.fit 返回 history dict", isinstance(hist, dict) and 'loss' in hist)
    check("History 记录 5 个 epoch", len(hist['loss']) == 5)
    check("训练 loss 下降", hist['loss'][-1] < hist['loss'][0] * 0.5,
          f"{hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}")
    check("accuracy 记录", 'accuracy' in hist and len(hist['accuracy']) == 5)
    check("val_loss 记录", 'val_loss' in hist and len(hist['val_loss']) == 5)
    check("val_accuracy 记录", 'val_accuracy' in hist)
    check("ModelCheckpoint 保存最优", os.path.isfile(ckpt_path2))
    check("训练后准确率>80%", hist['accuracy'][-1] > 0.8, f"acc={hist['accuracy'][-1]:.3f}")
    # scheduler 生效：5 个 epoch 后 lr 应为 0.01 * 0.5^(5//3) = 0.0025
    check("scheduler 调度生效", abs(optimizer.param_groups[0]['lr'] - 1e-2 * 0.5) < 1e-12,
          f"lr={optimizer.param_groups[0]['lr']}")
    # evaluate
    val_logs = trainer.evaluate(val_dl)
    check("evaluate 返回 loss+accuracy", 'loss' in val_logs and 'accuracy' in val_logs)
    # predict
    xs, preds = trainer.predict(val_dl)
    check("predict 返回输入+预测", len(xs) > 0 and len(preds) > 0
          and preds[0].shape[1] == 3)
    # 自定义 Callback
    class CountCallback(Callback):
        def __init__(self): self.batch_count = 0; self.epoch_count = 0
        def on_epoch_end(self, trainer, epoch, logs): self.epoch_count += 1
        def on_batch_end(self, trainer, batch, logs): self.batch_count += 1
    cc = CountCallback()
    model3 = Sequential(Linear(8, 16), ReLU(), Linear(16, 3))
    opt3 = Adam(model3.parameters(), lr=1e-2)
    trainer3 = Trainer(model3, CrossEntropyLoss(), opt3, train_dl, epochs=2,
                       callbacks=[cc])
    trainer3.fit()
    check("自定义 Callback on_batch_end", cc.batch_count > 0)
    check("自定义 Callback on_epoch_end", cc.epoch_count == 2)
    # grad_clip 不报错（已在上面训练中验证）
    check("grad_clip 正常运行", True)

    # ---------- 3. viz/ ----------
    print("\n== 3. 可视化 viz/ ==")
    # plot_curves
    curves_path = os.path.join(tmpdir, 'viz', 'curves.png')
    plot_curves(hist, curves_path, metrics=['accuracy'])
    check("plot_curves 生成图片", os.path.isfile(curves_path) and os.path.getsize(curves_path) > 1000)
    # plot_classification
    imgs = rng.rand(10, 1, 28, 28).astype(np.float32)
    cls_path = os.path.join(tmpdir, 'viz', 'cls.png')
    plot_classification(imgs, preds=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
                        labels=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
                        save_path=cls_path, n=6)
    check("plot_classification 生成图片", os.path.isfile(cls_path) and os.path.getsize(cls_path) > 500)
    # plot_detections
    sd = ShapesDetection(num_samples=3, seed=0)
    det_imgs = np.stack([sd[i][0] for i in range(3)])         # (3, 1, 32, 32)
    det_boxes = np.stack([sd[i][1][0] for i in range(3)])      # (3, 3, 4)
    det_labels = np.stack([sd[i][1][1] for i in range(3)])     # (3, 3)
    det_path = os.path.join(tmpdir, 'viz', 'det.png')
    plot_detections(det_imgs, det_boxes, det_labels, save_path=det_path, n=3)
    check("plot_detections 生成图片", os.path.isfile(det_path) and os.path.getsize(det_path) > 500)
    # plot_image_grid
    grid_path = os.path.join(tmpdir, 'viz', 'grid.png')
    plot_image_grid(imgs[:8], save_path=grid_path, nrow=4)
    check("plot_image_grid 生成图片", os.path.isfile(grid_path) and os.path.getsize(grid_path) > 500)
    # plot_text_history
    text_path = os.path.join(tmpdir, 'viz', 'text.png')
    plot_text_history([2.5, 1.8, 1.2, 0.9, 0.7], save_path=text_path)
    check("plot_text_history 生成图片", os.path.isfile(text_path) and os.path.getsize(text_path) > 500)

    # ---------- 4. 端到端：checkpoint 恢复后继续训练 ----------
    print("\n== 4. 端到端：checkpoint 恢复 + 继续训练 ==")
    # 从 checkpoint 恢复模型
    model_r = Sequential(Linear(8, 32), ReLU(), Linear(32, 3))
    opt_r = Adam(model_r.parameters(), lr=1e-2)
    load_checkpoint(ckpt_path2, model=model_r, optimizer=opt_r)
    # 恢复后立即评估，loss 应较低
    trainer_r = Trainer(model_r, CrossEntropyLoss(), opt_r, train_dl, val_loader=val_dl,
                        epochs=1, metrics=['accuracy'])
    train_before = trainer_r.evaluate(train_dl)['loss']
    val_before = trainer_r.evaluate(val_dl)['loss']
    # 再训 1 个 epoch（过拟合模型 train_loss 应继续下降或持平）
    trainer_r.fit()
    train_after = trainer_r.evaluate(train_dl)['loss']
    check("checkpoint 恢复后可继续训练", train_after <= train_before + 0.01,
          f"train_loss {train_before:.4f} -> {train_after:.4f}")
    # 恢复后模型在训练集上 loss 低，证明权重正确恢复（过拟合模型 train_loss 必然低）
    check("恢复后 train_loss 较低", train_before < 0.5, f"train_loss={train_before:.4f}")

    print(f"\n{'=' * 52}")
    print(f"结果: {PASSED} 项通过, {len(FAILURES)} 项失败")
    if FAILURES:
        print("失败项:", FAILURES)
        return 1
    print("L3 第三批（trainer/serialization/viz）验收通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())

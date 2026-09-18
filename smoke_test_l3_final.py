"""L3 收官 · 验收测试脚本（完整包入口 + DataParallel 教学版 + 端到端训练）

用法：
    python smoke_test_l3_final.py        # 与 zerotorch/ 目录同级，或位于其子目录

覆盖：
    1. 完整包入口：zt.nn / zt.optim / zt.loss / zt.data / zt.metrics /
                   zt.serialization / zt.viz / zt.parallel / zt.Trainer 等全部可访问
    2. DataParallel 教学版：包装模型 / forward / train_step / 参数委托 / is_data_parallel
    3. Trainer + DataParallel 端到端训练（走 train_step 路径，loss 下降）
    4. 完整 API 一行 import 后训练 MLP（验证包入口完整性）

退出码：全部通过返回 0，任一失败返回 1。
"""
import os
import sys

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

    # ---------- 0. 目录诊断 ----------
    print("\n== 0. 目录结构诊断 ==")
    cands = [os.path.join(_base, 'zerotorch'),
             os.path.join(os.path.dirname(_base), 'zerotorch')]
    pkg_dir = next((c for c in cands if os.path.isdir(c)), None)
    if pkg_dir is None:
        print(f"  [ERROR] 找不到包目录（已检查: {cands}）")
        return 1
    required = ['__init__.py', 'tensor.py', 'functions.py', 'metrics.py',
                'serialization.py', 'trainer.py',
                os.path.join('parallel', '__init__.py'), os.path.join('parallel', 'dataparallel.py')]
    missing = [f for f in required if not os.path.isfile(os.path.join(pkg_dir, f))]
    if missing:
        print(f"  [ERROR] 缺少文件: {missing}")
        return 1
    print("  [PASS] L3 收官目录结构完整")

    # ---------- 1. 完整包入口 ----------
    print("\n== 1. 完整包入口（一行 import 后全部 API 可用） ==")
    import zerotorch as zt
    check("zt.__version__", hasattr(zt, '__version__') and zt.__version__ == '0.1.0')
    check("zt.Tensor / zt.tensor", hasattr(zt, 'Tensor') and hasattr(zt, 'tensor'))
    check("zt.F", hasattr(zt, 'F'))
    check("zt.ops", hasattr(zt, 'ops'))
    check("zt.nn", hasattr(zt, 'nn') and hasattr(zt.nn, 'Linear') and hasattr(zt.nn, 'Sequential'))
    check("zt.loss", hasattr(zt, 'loss') and hasattr(zt.loss, 'CrossEntropyLoss'))
    check("zt.optim", hasattr(zt, 'optim') and hasattr(zt.optim, 'Adam') and hasattr(zt.optim, 'SGD'))
    check("zt.data", hasattr(zt, 'data') and hasattr(zt.data, 'DataLoader') and hasattr(zt.data, 'TensorDataset'))
    check("zt.metrics", hasattr(zt, 'metrics') and hasattr(zt.metrics, 'accuracy'))
    check("zt.serialization", hasattr(zt, 'serialization') and hasattr(zt.serialization, 'save_checkpoint'))
    check("zt.viz", hasattr(zt, 'viz') and hasattr(zt.viz, 'plot_curves'))
    check("zt.parallel", hasattr(zt, 'parallel') and hasattr(zt.parallel, 'DataParallel'))
    # 训练器顶层快捷类
    check("zt.Trainer", hasattr(zt, 'Trainer'))
    check("zt.Callback", hasattr(zt, 'Callback'))
    check("zt.History", hasattr(zt, 'History'))
    check("zt.ModelCheckpoint", hasattr(zt, 'ModelCheckpoint'))
    check("zt.ProgressBar", hasattr(zt, 'ProgressBar'))
    check("zt.DataParallel", hasattr(zt, 'DataParallel'))

    # ---------- 2. DataParallel 教学版 ----------
    print("\n== 2. DataParallel 教学版（单 CPU） ==")
    base_model = zt.nn.Sequential(zt.nn.Linear(8, 16), zt.nn.ReLU(), zt.nn.Linear(16, 3))
    dp = zt.DataParallel(base_model, num_workers=2)
    check("is_data_parallel=True", getattr(dp, 'is_data_parallel', False) is True)
    check("num_workers 记录", dp.num_workers == 2)
    # forward 委托
    x = zt.tensor(rng.randn(4, 8).astype(np.float64))
    out_dp = dp(x)
    out_base = base_model(x)
    check("forward 委托原模型", np.allclose(out_dp.data, out_base.data))
    # 参数委托
    check("parameters 委托", list(dp.parameters()) == list(base_model.parameters()))
    check("state_dict 委托", dp.state_dict().keys() == base_model.state_dict().keys())
    check("num_parameters 委托", dp.num_parameters() == base_model.num_parameters())
    # train/eval 委托
    dp.eval()
    check("eval 委托", base_model.training is False)
    dp.train()
    check("train 委托", base_model.training is True)
    # zero_grad 委托
    for p in dp.parameters():
        p.grad = np.ones_like(p.data)
    dp.zero_grad()
    check("zero_grad 委托", all(p.grad is None for p in base_model.parameters()))
    # train_step 本地执行（教学版核心）
    x2 = zt.tensor(rng.randn(4, 8).astype(np.float64))
    y2 = np.array([0, 1, 2, 0])
    loss_fn = zt.loss.CrossEntropyLoss()
    loss = dp.train_step(x2, y2, loss_fn)
    check("train_step 返回 loss", loss.data.ndim == 0 and np.isfinite(loss.data))
    check("train_step 产生梯度", all(p.grad is not None for p in base_model.parameters()))
    # 非 Module 报错
    try:
        zt.DataParallel("not_a_module")
        check("非 Module 报错", False)
    except TypeError:
        check("非 Module 报错", True)

    # ---------- 3. Trainer + DataParallel 端到端训练 ----------
    print("\n== 3. Trainer + DataParallel 端到端训练（走 train_step 路径） ==")
    from zerotorch.data.datasets import make_blobs
    Xtr, ytr = make_blobs(n_samples=300, n_features=8, centers=3, seed=1)
    train_dl = zt.data.DataLoader(zt.data.TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
    model_dp = zt.DataParallel(
        zt.nn.Sequential(zt.nn.Linear(8, 32), zt.nn.ReLU(), zt.nn.Linear(32, 3)),
        num_workers=2)
    opt = zt.optim.Adam(model_dp.parameters(), lr=1e-2)
    history_cb = zt.History()
    trainer = zt.Trainer(model_dp, loss_fn, opt, train_dl, epochs=5,
                         metrics=['accuracy'], callbacks=[history_cb, zt.ProgressBar()])
    hist = trainer.fit()
    check("DataParallel 训练 loss 下降", hist['loss'][-1] < hist['loss'][0] * 0.3,
          f"{hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}")
    check("DataParallel 训练 accuracy>80%", hist['accuracy'][-1] > 0.8,
          f"acc={hist['accuracy'][-1]:.3f}")
    # 确认走了 train_step 路径（is_data_parallel=True 时 Trainer._train_step 调用 dp.train_step）
    check("Trainer 识别 DataParallel", getattr(trainer.model, 'is_data_parallel', False) is True)

    # ---------- 4. 完整 API 一行 import 训练 ----------
    print("\n== 4. 完整 API 一行 import 后训练 MLP ==")
    # 全部通过 zt.xxx 访问，不做任何子模块 import
    net = zt.nn.Sequential(zt.nn.Linear(4, 16), zt.nn.ReLU(), zt.nn.Linear(16, 2))
    crit = zt.loss.CrossEntropyLoss()
    optimizer = zt.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    Xs = np.concatenate([rng.randn(50, 4) + 2.0, rng.randn(50, 4) - 2.0]).astype(np.float32)
    ys = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
    loader = zt.data.DataLoader(zt.data.TensorDataset(Xs, ys), batch_size=16, shuffle=True)
    losses = []
    for epoch in range(10):
        for xb, yb in loader:
            optimizer.zero_grad()
            l = crit(net(xb), yb.data)
            l.backward()
            optimizer.step()
        full_l = float(crit(net(zt.tensor(Xs)), ys).data)
        losses.append(full_l)
    check("完整 API 训练 loss 下降", losses[-1] < losses[0] * 0.3,
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    preds = np.argmax(net(zt.tensor(Xs)).data, axis=1)
    check("完整 API 训练准确率>90%", float(np.mean(preds == ys)) > 0.9,
          f"acc={float(np.mean(preds == ys)):.3f}")

    print(f"\n{'=' * 52}")
    print(f"结果: {PASSED} 项通过, {len(FAILURES)} 项失败")
    if FAILURES:
        print("失败项:", FAILURES)
        return 1
    print("L3 收官（完整包入口 + DataParallel 教学版）验收通过 ✅")
    print("L1+L2+L3 全部完成：自动微分 + nn层 + 训练闭环 + 数据 + 可视化 + 并行接口")
    return 0


if __name__ == "__main__":
    sys.exit(main())

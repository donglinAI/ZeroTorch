"""L1 自动微分核心 · 验收测试脚本（自包含，只依赖 L1 已交付文件）

用法：
    python smoke_test_l1.py

覆盖：
    1. 包结构与算子完整性（tensor + 47 个算子 + F API + gradcheck）
    2. 逐类算子数值梯度检查（gradcheck：标量输出）
    3. 非标量算子手写中心差分抽查（Matmul 广播 / Conv2d / Upsample）
    4. 端到端闭环：两层 MLP + CrossEntropy，loss 下降 + 梯度有限
    5. 可选：与 PyTorch 梯度对照（机器装有 torch 时自动执行）

退出码：全部通过返回 0，任一失败返回 1。
"""
import sys
import os

# 允许从脚本所在目录或项目根目录直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

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


def numerical_gradient(fn, x, *args):
    """中心差分（与 gradcheck 同款，支持非标量输出时自行对输出求和）。"""
    eps = 1e-6
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp = x.copy(); xp[idx] += eps
        xm = x.copy(); xm[idx] -= eps
        g[idx] = (fn(xp, *args) - fn(xm, *args)) / (2 * eps)
    return g


def main():
    rng = np.random.RandomState(42)

    # ---------- 0. 目录结构诊断 ----------
    print("\n== 0. 目录结构诊断 ==")
    _base = os.path.dirname(os.path.abspath(__file__))
    _cands = [os.path.join(_base, 'zerotorch'),          # 脚本同级（用户推荐布局）
              os.path.join(os.path.dirname(_base), 'zerotorch')]  # 上级（脚本在子目录时）
    pkg_dir = next((c for c in _cands if os.path.isdir(c)), None)
    if pkg_dir is None:
        print(f"  [ERROR] 找不到包目录（已检查: {_cands}）")
        print("  请确认 smoke_test_l1.py 与 zerotorch/ 目录同级，例如：")
        print("    ZeroTorch/")
        print("    ├── zerotorch/")
        print("    │   ├── __init__.py     ← 必须有（见下方说明）")
        print("    │   ├── tensor.py")
        print("    │   ├── functions.py")
        print("    │   ├── ops/")
        print("    │   └── utils/gradcheck.py")
        print("    └── smoke_test_l1.py")
        return 1
    required = ['__init__.py', 'tensor.py', 'functions.py',
                os.path.join('ops', '__init__.py'), os.path.join('ops', 'basic.py'),
                os.path.join('ops', 'matrix.py'), os.path.join('ops', 'activations.py'),
                os.path.join('ops', 'shape.py'), os.path.join('ops', 'conv.py'),
                os.path.join('ops', 'norm.py'), os.path.join('ops', 'losses.py'),
                os.path.join('ops', 'upsample.py'),
                os.path.join('utils', 'gradcheck.py')]
    missing = [r for r in required if not os.path.isfile(os.path.join(pkg_dir, r))]
    if missing:
        print(f"  [ERROR] zerotorch/ 目录缺少以下文件：")
        for m in missing:
            print(f"    - {m}")
        if '__init__.py' in missing:
            print("  >>> 缺少 __init__.py 是当前报错（no attribute '__version__'）的直接原因。")
            print("  >>> 请把交付的 L1 版 __init__.py 放到 zerotorch/ 目录下。")
        return 1
    print("  [PASS] zerotorch/ 目录结构完整（%d 个必需文件齐）" % len(required))

    # ---------- 1. 结构完整性 ----------
    print("\n== 1. 包结构与算子完整性 ==")
    import zerotorch as zt
    ver = getattr(zt, '__version__', '未定义')
    check("import zerotorch", True, f"版本 {ver}")
    check("Tensor / tensor 存在", hasattr(zt, "Tensor") and callable(zt.tensor))
    check("F API 存在", hasattr(zt, "F"))
    check("ops 算子总数 = 47", len(zt.ops.__all__) == 47, f"实际 {len(zt.ops.__all__)}")
    from zerotorch.utils.gradcheck import gradcheck
    check("gradcheck 工具可导入", True)

    # ---------- 2. gradcheck 逐类算子（标量化输出） ----------
    print("\n== 2. 标量算子 gradcheck ==")
    x = zt.tensor(rng.randn(4, 4).astype(np.float64), requires_grad=True)

    def gd(name, fn, inputs):
        ok, err, _ = gradcheck(fn, inputs)
        check(name, ok, f"相对误差 {err:.2e}")

    # 四则/超越
    gd("Add", lambda a, b: zt.F.sum(zt.F.add(a, b)),
       [zt.tensor(rng.randn(3, 1).astype(np.float64), requires_grad=True),
        zt.tensor(rng.randn(5).astype(np.float64), requires_grad=True)])
    gd("Mul", lambda a, b: zt.F.sum(zt.F.mul(a, b)),
       [zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True),
        zt.tensor(rng.randn(4).astype(np.float64), requires_grad=True)])
    gd("Div", lambda a, b: zt.F.sum(zt.F.div(a + 2.0, b + 3.0)),
       [zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True),
        zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True)])
    gd("Pow", lambda a: zt.F.sum(zt.F.pow_(a + 2.0, 3)),
       [zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True)])
    gd("Exp/Log/Sqrt", lambda a: zt.F.sum(zt.F.sqrt(zt.F.exp(zt.F.log(a + 3.0)))),
       [zt.tensor(rng.randn(3, 4).astype(np.float64) + 1.0, requires_grad=True)])
    gd("Abs", lambda a: zt.F.sum(zt.F.abs_(a)),
       [zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True)])
    # 激活
    for name, fn in [("Relu", zt.F.relu), ("Sigmoid", zt.F.sigmoid),
                     ("Tanh", zt.F.tanh), ("Gelu", zt.F.gelu),
                     ("Silu", zt.F.silu), ("Softmax", zt.F.softmax),
                     ("LogSoftmax", zt.F.log_softmax),
                     ("LeakyRelu", lambda a: zt.F.leaky_relu(a, 0.1))]:
        gd(name, lambda a, _fn=fn: zt.F.sum(_fn(a)),
           [zt.tensor(rng.randn(3, 5).astype(np.float64), requires_grad=True)])
    # 形状/规约
    gd("Sum(dim)", lambda a: zt.F.sum(zt.F.sum(a, dim=1, keepdims=False)),
       [zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True)])
    gd("Mean", lambda a: zt.F.mean(a),
       [zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True)])
    gd("Matmul(2D)", lambda a, b: zt.F.sum(zt.F.matmul(a, b)),
       [zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True),
        zt.tensor(rng.randn(4, 5).astype(np.float64), requires_grad=True)])
    gd("Reshape", lambda a: zt.F.sum(zt.F.reshape(a, (4, 3))),
       [zt.tensor(rng.randn(3, 4).astype(np.float64), requires_grad=True)])
    gd("BroadcastTo", lambda a: zt.F.sum(zt.F.broadcast_to(a, (3, 4))),
       [zt.tensor(rng.randn(1, 4).astype(np.float64), requires_grad=True)])
    # 损失（融合算子，标量输出，最关键的检查）
    logits = rng.randn(6, 5).astype(np.float64)
    tgt = rng.randint(0, 5, 6)
    gd("CrossEntropy", lambda L: zt.F.cross_entropy(L, tgt),
       [zt.tensor(logits, requires_grad=True)])
    bce_t = (rng.rand(4, 3) > 0.5).astype(np.float64)
    gd("BCEWithLogits", lambda L: zt.F.binary_cross_entropy_with_logits(L, bce_t),
       [zt.tensor(rng.randn(4, 3).astype(np.float64), requires_grad=True)])
    sl1_tgt = rng.randn(4, 4).astype(np.float64)
    gd("SmoothL1", lambda P: zt.F.smooth_l1_loss(P, sl1_tgt),
       [zt.tensor(rng.randn(4, 4).astype(np.float64), requires_grad=True)])

    # ---------- 3. 非标量算子中心差分抽查 ----------
    print("\n== 3. 非标量算子中心差分抽查 ==")
    # Matmul 广播批量 (3,4) @ (2,4,5)
    a = rng.randn(3, 4).astype(np.float64)
    b = rng.randn(2, 4, 5).astype(np.float64)
    ta = zt.tensor(a, requires_grad=True)
    tb = zt.tensor(b, requires_grad=True)
    zt.F.matmul(ta, tb).backward(np.ones((2, 3, 5)))
    ga = numerical_gradient(lambda X: zt.F.matmul(zt.tensor(X), zt.tensor(b)).data.sum(), a)
    check("Matmul 广播批量", np.max(np.abs(ga - ta.grad)) < 1e-7,
          f"dA 误差 {np.max(np.abs(ga - ta.grad)):.2e}")
    # Conv2d（im2col）
    xc = rng.randn(1, 2, 6, 6).astype(np.float64)
    wc = rng.randn(3, 2, 3, 3).astype(np.float64) * 0.5
    tc = zt.tensor(xc, requires_grad=True)
    tw = zt.tensor(wc, requires_grad=True)
    zt.F.conv2d(tc, tw, None, 1, 1).backward(np.ones((1, 3, 6, 6)))
    gx = numerical_gradient(lambda X: zt.F.conv2d(zt.tensor(X), zt.tensor(wc), None, 1, 1).data.sum(), xc)
    check("Conv2d", np.max(np.abs(gx - tc.grad)) < 1e-7,
          f"dX 误差 {np.max(np.abs(gx - tc.grad)):.2e}")
    # MaxPool2d
    xp_ = rng.randn(1, 1, 6, 6).astype(np.float64)
    tp = zt.tensor(xp_, requires_grad=True)
    zt.F.max_pool2d(tp, 2, 2).backward(np.ones((1, 1, 3, 3)))
    gp = numerical_gradient(lambda X: zt.F.max_pool2d(zt.tensor(X), 2, 2).data.sum(), xp_)
    check("MaxPool2d", np.max(np.abs(gp - tp.grad)) < 1e-7,
          f"误差 {np.max(np.abs(gp - tp.grad)):.2e}")
    # LayerNorm / BatchNorm
    xl = rng.randn(4, 8).astype(np.float64)
    gm = rng.randn(8).astype(np.float64)
    be = rng.randn(8).astype(np.float64)
    tl = zt.tensor(xl, requires_grad=True)
    zt.F.layer_norm(tl, zt.tensor(gm), zt.tensor(be)).backward(np.ones((4, 8)))
    gl = numerical_gradient(lambda X: zt.F.layer_norm(zt.tensor(X), zt.tensor(gm), zt.tensor(be)).data.sum(), xl)
    check("LayerNorm", np.max(np.abs(gl - tl.grad)) < 1e-6,
          f"误差 {np.max(np.abs(gl - tl.grad)):.2e}")
    # Upsample（扩展算子）
    xu = rng.randn(1, 1, 3, 3).astype(np.float64)
    tu = zt.tensor(xu, requires_grad=True)
    zt.ops.Upsample.apply(tu, 2).backward(np.ones((1, 1, 6, 6)))
    gu = numerical_gradient(lambda X: zt.ops.Upsample.apply(zt.tensor(X), 2).data.sum(), xu)
    check("Upsample", np.max(np.abs(gu - tu.grad)) < 1e-7,
          f"误差 {np.max(np.abs(gu - tu.grad)):.2e}")

    # ---------- 4. 端到端闭环：两层 MLP 训练 ----------
    print("\n== 4. 端到端闭环：两层 MLP + CrossEntropy ==")
    N, D, H, C = 64, 8, 16, 3
    X = rng.randn(N, D).astype(np.float64)
    Y = rng.randint(0, C, N)
    w1 = zt.tensor(rng.randn(D, H).astype(np.float64) * 0.5, requires_grad=True)
    b1 = zt.tensor(np.zeros(H), requires_grad=True)
    w2 = zt.tensor(rng.randn(H, C).astype(np.float64) * 0.5, requires_grad=True)
    b2 = zt.tensor(np.zeros(C), requires_grad=True)

    def step(lr=0.1):
        h = zt.F.relu(zt.F.matmul(zt.tensor(X), w1) + b1)
        logits = zt.F.matmul(h, w2) + b2
        loss = zt.F.cross_entropy(logits, Y)
        loss.backward()
        grads = [w1.grad, b1.grad, w2.grad, b2.grad]
        ok_finite = all(np.isfinite(g).all() for g in grads)
        ok_shape = [g.shape for g in grads] == [(D, H), (H,), (H, C), (C,)]
        for p in (w1, b1, w2, b2):
            p.data -= lr * p.grad
            p.zero_grad()
        return float(loss.data), ok_finite and ok_shape

    l0, ok0 = step()
    losses = [l0]
    for _ in range(19):
        l, ok = step()
        losses.append(l)
        ok0 = ok0 and ok
    check("梯度形状+有限", ok0)
    check("loss 单调下降", losses[-1] < losses[0] * 0.5,
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"    loss 轨迹: " + " -> ".join(f"{v:.3f}" for v in losses[::5]))

    # ---------- 5. 可选：与 PyTorch 梯度对照 ----------
    print("\n== 5. 与 PyTorch 梯度对照（可选） ==")
    try:
        import torch
        torch.manual_seed(42)
        # 第 4 节训练循环末尾已 zero_grad，这里重新前向反向收集 zerotorch 梯度
        h = zt.F.relu(zt.F.matmul(zt.tensor(X), w1) + b1)
        logits = zt.F.matmul(h, w2) + b2
        zt.F.cross_entropy(logits, Y).backward()
        xt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        w1t = torch.tensor(w1.data.copy(), dtype=torch.float64, requires_grad=True)
        b1t = torch.tensor(b1.data.copy(), dtype=torch.float64, requires_grad=True)
        w2t = torch.tensor(w2.data.copy(), dtype=torch.float64, requires_grad=True)
        b2t = torch.tensor(b2.data.copy(), dtype=torch.float64, requires_grad=True)
        ht = torch.relu(xt @ w1t + b1t)
        loss_t = torch.nn.functional.cross_entropy(ht @ w2t + b2t, torch.tensor(Y))
        loss_t.backward()
        assert w1.grad is not None, "zerotorch 梯度未收集（内部错误）"
        d_w1 = np.abs(w1t.grad.numpy() - w1.grad).max()
        d_b1 = np.abs(b1t.grad.numpy() - b1.grad).max()
        d_w2 = np.abs(w2t.grad.numpy() - w2.grad).max()
        d_b2 = np.abs(b2t.grad.numpy() - b2.grad).max()
        check("PyTorch 梯度对照", max(d_w1, d_b1, d_w2, d_b2) < 1e-8,
              f"最大梯度差 {max(d_w1, d_b1, d_w2, d_b2):.2e}")
    except ImportError:
        print("  (未安装 torch，跳过对照)")

    # ---------- 汇总 ----------
    print(f"\n{'='*52}")
    print(f"结果: {PASSED} 项通过, {len(FAILURES)} 项失败")
    if FAILURES:
        print("失败项:", FAILURES)
        return 1
    print("L1 自动微分核心验收通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())

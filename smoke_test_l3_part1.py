"""L3 第一批 · 验收测试脚本（loss/ + optim/ + metrics.py）

用法：
    python smoke_test_l3_part1.py        # 与 zerotorch/ 目录同级，或位于其子目录

覆盖：
    1. loss/：7 个损失前向标量 + 反向梯度有限 + list/ndarray 一致 + 注册机制 + 即插即用
    2. optim/：SGD 手推 / Adam step+zero_grad / AdamW 解耦权重衰减 /
               state_dict 往返 / StepLR / CosineAnnealing / clip_grad_norm_ / 注册
    3. metrics.py：accuracy / iou / mean_iou / 注册机制
    4. 端到端：Adam + CrossEntropyLoss 训练 MLP，loss 显著下降
    5. 可选：与 PyTorch 对照（同构网络 + 相同初始权重，Adam 单步参数更新量一致）

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
    required = {
        'loss': ['__init__.py', 'losses.py'],
        'optim': ['__init__.py', 'base.py', 'optimizers.py', 'lr_scheduler.py'],
    }
    for sub, files in required.items():
        missing = [f for f in files if not os.path.isfile(os.path.join(pkg_dir, sub, f))]
        if missing:
            print(f"  [ERROR] {sub}/ 缺少文件: {missing}")
            return 1
    if not os.path.isfile(os.path.join(pkg_dir, 'metrics.py')):
        print("  [ERROR] 缺少 metrics.py")
        return 1
    print("  [PASS] loss/ optim/ metrics.py 目录结构完整")

    import zerotorch as zt
    from zerotorch.loss import (MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss,
                                 BCEWithLogitsLoss, BCELoss, KLGaussianLoss,
                                 Loss, register_loss, LOSS_REGISTRY)
    from zerotorch.optim import (SGD, Adam, AdamW, Optimizer, register_optimizer,
                                  OPTIM_REGISTRY, clip_grad_norm_,
                                  StepLR, CosineAnnealingLR, LambdaLR)
    from zerotorch.metrics import accuracy, iou, mean_iou, METRIC_REGISTRY, register_metric
    from zerotorch.nn.container import Sequential
    from zerotorch.nn.layers import Linear, ReLU

    # ---------- 1. loss/ ----------
    print("\n== 1. 损失函数 loss/ ==")
    logits = zt.tensor(rng.randn(8, 3).astype(np.float64), requires_grad=True)
    labels_list = [0, 1, 2, 0, 1, 2, 0, 1]      # 故意用 list 触发 np.asarray 修复
    labels_arr = np.array(labels_list)

    for name, cls in [('MSE', MSELoss), ('L1', L1Loss), ('SmoothL1', SmoothL1Loss),
                      ('CrossEntropy', CrossEntropyLoss),
                      ('BCEWithLogits', BCEWithLogitsLoss)]:
        loss_fn = cls()
        if name in ('MSE', 'L1', 'SmoothL1'):
            tgt = zt.tensor(rng.randn(8, 3).astype(np.float64))
            l = loss_fn(logits, tgt)
        elif name == 'BCEWithLogits':
            tgt = zt.tensor(rng.randint(0, 2, (8, 3)).astype(np.float64))
            l = loss_fn(logits, tgt)
        else:
            l = loss_fn(logits, labels_list)
        l.backward()
        check(f"{name} 标量+梯度有限",
              l.data.ndim == 0 and logits.grad is not None and np.isfinite(logits.grad).all(),
              f"loss={float(l.data):.4f}")
        logits.grad = None

    # CrossEntropy list 与 ndarray 结果一致（修复点验证）
    l_list = CrossEntropyLoss()(logits, labels_list)
    l_arr = CrossEntropyLoss()(logits, labels_arr)
    check("CrossEntropy list==ndarray",
          abs(float(l_list.data) - float(l_arr.data)) < 1e-12,
          f"list={float(l_list.data):.6f} ndarray={float(l_arr.data):.6f}")

    # BCELoss（概率形式）
    prob = zt.tensor(np.clip(rng.rand(8, 3), 0.01, 0.99), requires_grad=True)
    bce_tgt = zt.tensor(rng.randint(0, 2, (8, 3)).astype(np.float64))
    l_bce = BCELoss()(prob, bce_tgt); l_bce.backward()
    check("BCELoss 标量+梯度", l_bce.data.ndim == 0 and prob.grad is not None)

    # KLGaussian（VAE 用）
    mu = zt.tensor(rng.randn(4, 8).astype(np.float64), requires_grad=True)
    lv = zt.tensor(rng.randn(4, 8).astype(np.float64), requires_grad=True)
    kl = KLGaussianLoss()(mu, lv); kl.backward()
    check("KLGaussian 标量+梯度", kl.data.ndim == 0 and mu.grad is not None and lv.grad is not None)

    # 注册机制
    check("LOSS_REGISTRY 含 7 个", len(LOSS_REGISTRY) == 7, str(sorted(LOSS_REGISTRY)))

    # 即插即用：自定义损失
    @register_loss('my_abs_loss')
    class MyAbsLoss(Loss):
        def forward(self, pred, target):
            return zt.F.abs_(pred - target).mean()
    check("自定义损失即插即用", 'my_abs_loss' in LOSS_REGISTRY
          and LOSS_REGISTRY['my_abs_loss'] is MyAbsLoss)
    my_l = MyAbsLoss()(zt.tensor(np.array([1.0, 2.0])), zt.tensor(np.array([0.0, 0.0])))
    check("自定义损失前向", abs(float(my_l.data) - 1.5) < 1e-9, f"={float(my_l.data)}")

    # ---------- 2. optim/ ----------
    print("\n== 2. 优化器 optim/ ==")
    # SGD 手推
    w = zt.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    w.grad = np.array([0.1, 0.2, 0.3])
    SGD([w], lr=0.1).step()
    check("SGD 手推", np.allclose(w.data, [0.99, 1.98, 2.97]), str(w.data))

    # SGD + momentum
    w2 = zt.tensor(np.array([1.0]), requires_grad=True)
    w2.grad = np.array([1.0])
    opt_m = SGD([w2], lr=0.1, momentum=0.9)
    opt_m.step()   # buf=1.0, g=buf=1.0, w=1-0.1*1=0.9
    check("SGD+momentum 第一步", np.allclose(w2.data, [0.9]), str(w2.data))
    w2.grad = np.array([1.0])
    opt_m.step()   # buf=0.9*1+1=1.9, g=1.9, w=0.9-0.19=0.71
    check("SGD+momentum 第二步", np.allclose(w2.data, [0.71]), str(w2.data))

    # Adam step + zero_grad
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    opt = Adam(model.parameters(), lr=1e-3)
    loss = zt.F.cross_entropy(model(zt.tensor(rng.randn(4, 4).astype(np.float64))),
                               [0, 1, 2, 0])
    loss.backward()
    before = [p.data.copy() for p in model.parameters()]
    opt.step(); opt.zero_grad()
    check("Adam step 更新参数", any(not np.array_equal(a, b)
                                     for a, b in zip(before, model.parameters())))
    check("Adam zero_grad 清空", all(p.grad is None for p in model.parameters()))

    # AdamW 解耦权重衰减：第一步 m=v=0，只有权重衰减生效
    w3 = zt.tensor(np.array([2.0, 4.0]), requires_grad=True)
    w3.grad = np.zeros(2)
    AdamW([w3], lr=0.1, weight_decay=0.1).step()
    check("AdamW 解耦权重衰减", np.allclose(w3.data, [1.98, 3.96]), str(w3.data))

    # Adam vs AdamW 权重衰减区别验证
    # 相同梯度+权重，Adam 把 wd 加进梯度（受 m/v 调制），AdamW 直接衰减参数
    wa = zt.tensor(np.array([3.0]), requires_grad=True); wa.grad = np.array([0.5])
    wb = zt.tensor(np.array([3.0]), requires_grad=True); wb.grad = np.array([0.5])
    Adam([wa], lr=0.1, weight_decay=0.1).step()    # g=0.5+0.1*3=0.8; m_hat=0.8; v_hat=0.64; update=0.1; w=2.9
    AdamW([wb], lr=0.1, weight_decay=0.1).step()   # 解耦: w=3-0.03=2.97; 梯度: m_hat=0.5,v_hat=0.25,update=0.1; w=2.87
    check("Adam vs AdamW 衰减行为不同",
          np.allclose(wa.data, [2.9]) and np.allclose(wb.data, [2.87]),
          f"Adam={wa.data} AdamW={wb.data}")

    # state_dict / load_state_dict 往返
    sd = opt.state_dict()
    opt2 = Adam(model.parameters(), lr=1e-3)
    opt2.load_state_dict(sd)
    check("optim state_dict 往返", opt2._step_count == opt._step_count
          and all(np.array_equal(opt2.state[id(p)].get('m', np.zeros(1)),
                                  opt.state[id(p)].get('m', np.zeros(1)))
                  for p in model.parameters()))

    # StepLR
    m4 = Sequential(Linear(2, 2))
    o4 = SGD(m4.parameters(), lr=0.1)
    sch = StepLR(o4, step_size=2, gamma=0.5)
    lrs = [o4.param_groups[0]['lr']]
    for _ in range(5):
        sch.step(); lrs.append(o4.param_groups[0]['lr'])
    check("StepLR 调度", lrs == [0.1, 0.1, 0.05, 0.05, 0.025, 0.025], str(lrs))

    # CosineAnnealingLR
    o5 = SGD(m4.parameters(), lr=0.1)
    sch2 = CosineAnnealingLR(o5, T_max=10, eta_min=0.0)
    for _ in range(10):
        sch2.step()
    check("Cosine 末点≈0", abs(o5.param_groups[0]['lr']) < 1e-6,
          f"lr={o5.param_groups[0]['lr']:.2e}")

    # LambdaLR（__init__ 已调用一次 step，last_epoch=0；再调 3 次到第 3 步）
    o6 = SGD(m4.parameters(), lr=0.1)
    sch3 = LambdaLR(o6, lr_lambda=lambda e: 0.5 ** e)
    for _ in range(3):
        sch3.step()
    check("LambdaLR 第3步", abs(o6.param_groups[0]['lr'] - 0.1 * 0.5 ** 3) < 1e-12,
          f"lr={o6.param_groups[0]['lr']}")

    # clip_grad_norm_
    w4 = zt.tensor(np.array([3.0, 4.0]), requires_grad=True)
    w4.grad = np.array([3.0, 4.0])
    total = clip_grad_norm_([w4], max_norm=1.0)
    check("clip_grad_norm_", abs(total - 5.0) < 1e-9 and np.allclose(w4.grad, [0.6, 0.8]),
          f"total={total} grad={w4.grad}")

    # 注册机制
    check("OPTIM_REGISTRY 含 3 个", sorted(OPTIM_REGISTRY) == ['adam', 'adamw', 'sgd'])

    # ---------- 3. metrics.py ----------
    print("\n== 3. 评估指标 metrics.py ==")
    logits2 = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    targets2 = np.array([0, 1, 0])
    check("accuracy 全对", abs(accuracy(logits2, targets2) - 1.0) < 1e-9)
    check("accuracy 半对", abs(accuracy(logits2, np.array([0, 0, 0])) - 2 / 3) < 1e-9)

    b = np.array([0.5, 0.5, 0.4, 0.4])
    check("iou 相同框=1", abs(iou(b, b) - 1.0) < 1e-9)
    check("iou 不相交=0", iou(np.array([0.1, 0.1, 0.1, 0.1]),
                                np.array([0.9, 0.9, 0.1, 0.1])) < 1e-9)
    # 部分重叠：b1 xyxy=[0.2,0.3,0.6,0.7], b2 xyxy=[0.4,0.3,0.8,0.7]
    # 交集 [0.4,0.3,0.6,0.7] = 0.2*0.4=0.08；并集 0.16+0.16-0.08=0.24；iou=1/3
    b1 = np.array([0.4, 0.5, 0.4, 0.4])
    b2 = np.array([0.6, 0.5, 0.4, 0.4])
    expected_iou = 1.0 / 3.0
    check("iou 部分重叠", abs(iou(b1, b2) - expected_iou) < 1e-9,
          f"iou={iou(b1, b2):.4f} expected={expected_iou:.4f}")

    pb = np.stack([b, b])[None]
    gb = np.stack([b, b])[None]
    check("mean_iou 相同=1", abs(mean_iou(pb, gb) - 1.0) < 1e-9)
    check("METRIC_REGISTRY", 'accuracy' in METRIC_REGISTRY and 'mean_iou' in METRIC_REGISTRY)

    # 即插即用：自定义指标
    @register_metric('top2_accuracy')
    def top2_accuracy(logits, targets):
        top2 = np.argsort(np.asarray(logits), axis=-1)[:, -2:]
        return float(np.mean([t in row for t, row in zip(targets, top2)]))
    check("自定义指标即插即用", 'top2_accuracy' in METRIC_REGISTRY)

    # ---------- 4. 端到端：Adam + CrossEntropyLoss 训练 ----------
    print("\n== 4. 端到端：Adam + CrossEntropyLoss 训练 ==")
    D = 8
    X = np.concatenate([rng.randn(64, D) + c
                         for c in [(-2.0,) * D, (0.0,) * D, (2.0,) * D]])
    Y = np.repeat([0, 1, 2], 64)
    net = Sequential(Linear(D, 16), ReLU(), Linear(16, 3))
    loss_fn = CrossEntropyLoss()
    opt = Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        l = loss_fn(net(zt.tensor(X)), Y)
        l.backward()
        opt.step()
        losses.append(float(l.data))
    check("Adam 训练 loss 显著下降", losses[-1] < losses[0] * 0.3,
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"    loss 轨迹: " + " -> ".join(f"{v:.3f}" for v in losses[::10]))
    # 训练后准确率
    final_acc = accuracy(net(zt.tensor(X)).data, Y)
    check("训练后准确率>90%", final_acc > 0.9, f"acc={final_acc:.3f}")

    # ---------- 5. 可选：与 PyTorch 对照 ----------
    print("\n== 5. 与 PyTorch 对照（可选） ==")
    try:
        import torch
        torch.manual_seed(0)
        # 同构网络 + 相同初始权重，Adam 单步参数更新量对照
        net_t = torch.nn.Sequential(
            torch.nn.Linear(D, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)).double()
        with torch.no_grad():
            for (n1, p1), (n2, p2) in zip(net.named_parameters(), net_t.named_parameters()):
                p2.copy_(torch.tensor(p1.data))
        opt_t = torch.optim.Adam(net_t.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
        opt_z = Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

        # 前向 + 反向
        loss_z = CrossEntropyLoss()(net(zt.tensor(X)), Y)
        loss_z.backward()
        loss_t = torch.nn.functional.cross_entropy(
            net_t(torch.tensor(X, dtype=torch.float64)), torch.tensor(Y))
        loss_t.backward()
        check("loss 值与 PyTorch 一致",
              abs(float(loss_z.data) - loss_t.item()) < 1e-8,
              f"zt={float(loss_z.data):.8f} torch={loss_t.item():.8f}")
        # step
        opt_z.step()
        opt_t.step()
        diffs = []
        for (n1, p1), (n2, p2) in zip(net.named_parameters(), net_t.named_parameters()):
            diffs.append(np.abs(p1.data - p2.detach().numpy()).max())
        check("Adam 单步参数更新与 PyTorch 一致", max(diffs) < 1e-10,
              f"最大参数差 {max(diffs):.2e}")
    except ImportError:
        print("  (未安装 torch，跳过对照)")

    print(f"\n{'=' * 52}")
    print(f"结果: {PASSED} 项通过, {len(FAILURES)} 项失败")
    if FAILURES:
        print("失败项:", FAILURES)
        return 1
    print("L3 第一批（loss/optim/metrics）验收通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())

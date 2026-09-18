"""L2 nn 层 · 验收测试脚本（自包含，依赖 L1 + nn/ 全部文件）

用法：
    python smoke_test_l2.py        # 与 zerotorch/ 目录同级，或位于其子目录

覆盖：
    1. Module/Parameter 基类：自动注册、递归命名、state_dict/load_state_dict、train/eval、zero_grad
    2. init：fan 计算、Xavier/Kaiming 分布统计
    3. layers：Linear/Conv2d/Embedding/BatchNorm1d/LayerNorm/Dropout/池化/Flatten/激活
       —— 前向形状 + 反向梯度 + 手推对照
    4. container：Sequential/ModuleList
    5. rnn：RNN/LSTM 前向形状 + 单步手推对照 + 梯度有限
    6. 端到端：nn.Sequential 搭 MLP 训练 30 步，loss 单调下降
    7. 可选：与 PyTorch 的 nn.Sequential 梯度对照（机器装有 torch 时自动执行）

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
        print("  请确认 smoke_test_l2.py 与 zerotorch/ 同级或在其子目录。")
        return 1
    nn_dir = os.path.join(pkg_dir, 'nn')
    nn_required = ['__init__.py', 'module.py', 'init.py', 'layers.py',
                   'container.py', 'rnn.py']
    missing = [f for f in nn_required if not os.path.isfile(os.path.join(nn_dir, f))]
    if missing:
        print("  [ERROR] nn/ 目录缺少文件:", missing)
        return 1
    print("  [PASS] nn/ 目录结构完整（6 个必需文件齐）")

    import zerotorch as zt
    from zerotorch.nn.module import Module, Parameter
    from zerotorch.nn import init

    # ---------- 1. Module/Parameter 基类 ----------
    print("\n== 1. Module / Parameter 基类 ==")
    class Sub(Module):
        def __init__(self):
            super().__init__()
            self.b = Parameter(np.zeros(4))

    class Net(Module):
        def __init__(self):
            super().__init__()
            self.w = Parameter(np.random.randn(3, 4))
            self.sub = Sub()
            self.register_buffer('run_mean', np.zeros(3))
            self.scalar = 1.0          # 普通属性，不应进参数树

        def forward(self, x):
            return x @ self.w

    net = Net()
    names = [n for n, _ in net.named_parameters()]
    check("自动注册+嵌套命名", names == ['w', 'sub.b'], str(names))
    check("Parameter 叶子+requires_grad", net.w.requires_grad and net.w.is_leaf)
    check("普通属性不注册", 'scalar' not in dict(net.named_parameters()))
    # state_dict 往返
    sd = net.state_dict()
    check("state_dict 含参数+buffer", sorted(sd.keys()) == ['run_mean', 'sub.b', 'w'])
    net2 = Net()
    net2.load_state_dict(sd)
    check("load_state_dict 往返一致",
          np.array_equal(net2.w.data, net.w.data) and
          np.array_equal(net2.sub.b.data, net.sub.b.data))
    # strict 模式
    try:
        net2.load_state_dict({'w': np.zeros((3, 4))})
        check("strict 报错", False)
    except RuntimeError:
        check("strict 报错", True)
    # train/eval 递归
    net.eval()
    check("eval 递归", not net.training and not net.sub.training)
    net.train()
    check("train 递归", net.training and net.sub.training)
    # zero_grad / num_parameters
    out = net(zt.tensor(rng.randn(2, 3).astype(np.float64)))
    out.sum().backward()
    check("num_parameters", net.num_parameters() == 3 * 4 + 4, str(net.num_parameters()))
    net.zero_grad()
    check("zero_grad 清空", all(p.grad is None for p in net.parameters()))

    # ---------- 2. init ----------
    print("\n== 2. 参数初始化 ==")
    w = Parameter(np.empty((5, 3)))
    init.kaiming_uniform_(w, a=np.sqrt(5))
    bound = np.sqrt(6.0 / (1 + 5) / 3)
    check("kaiming_uniform 界内", np.abs(w.data).max() <= bound + 1e-9,
          f"max|w|={np.abs(w.data).max():.3f} bound={bound:.3f}")
    w2 = Parameter(np.empty((4, 2, 3, 3)))
    init.xavier_uniform_(w2)
    check("xavier_uniform 形状", w2.data.shape == (4, 2, 3, 3))
    init.zeros_(w)
    check("zeros_ 生效", (w.data == 0).all())

    # ---------- 3. layers ----------
    print("\n== 3. 网络层 layers ==")
    from zerotorch.nn.layers import (Linear, Conv2d, Embedding, Dropout,
                                     LayerNorm, BatchNorm1d, MaxPool2d, AvgPool2d,
                                     Flatten, ReLU, GELU)
    # Linear 手推
    x = rng.randn(5, 3).astype(np.float64)
    lin = Linear(3, 4)
    lin.weight.data = rng.randn(4, 3).astype(np.float64) * 0.5
    lin.bias.data = rng.randn(4).astype(np.float64) * 0.1
    lin(zt.tensor(x)).sum().backward()
    g = np.ones((5, 4))
    check("Linear dW 手推", np.allclose(lin.weight.grad, g.T @ x, atol=1e-10))
    check("Linear db 手推", np.allclose(lin.bias.grad, g.sum(0), atol=1e-10))
    # Conv2d
    xc = rng.randn(2, 3, 8, 8).astype(np.float64)
    conv = Conv2d(3, 5, 3, stride=2, padding=1)
    conv(zt.tensor(xc)).sum().backward()
    check("Conv2d 输出形状", conv(zt.tensor(xc)).data.shape == (2, 5, 4, 4))
    check("Conv2d 梯度形状", conv.weight.grad.shape == (5, 3, 3, 3)
          and conv.bias.grad.shape == (5,))
    # Embedding 梯度散射
    emb = Embedding(10, 6)
    emb.weight.data = rng.randn(10, 6).astype(np.float64)
    emb(np.array([3, 3, 7])).sum().backward()
    check("Embedding 重复索引累加",
          np.array_equal(emb.weight.grad[3], 2 * np.ones(6)) and
          np.array_equal(emb.weight.grad[7], np.ones(6)))
    # LayerNorm / BatchNorm1d
    ln = LayerNorm(6)
    ln(zt.tensor(rng.randn(4, 6).astype(np.float64))).sum().backward()
    check("LayerNorm 梯度形状", ln.weight.grad.shape == (6,) and ln.bias.grad.shape == (6,))
    bn = BatchNorm1d(4)
    bn(zt.tensor(rng.randn(2, 4, 3).astype(np.float64))).sum().backward()
    check("BatchNorm1d 梯度形状", bn.weight.grad.shape == (4,))
    rm0 = bn.running_mean.copy()
    bn.eval()
    bn(zt.tensor(rng.randn(2, 4, 3).astype(np.float64)))
    check("BatchNorm eval 不更新 running",
          np.array_equal(bn.running_mean, rm0))
    # Dropout train/eval
    dp = Dropout(0.5)
    dp.eval()
    check("Dropout eval 恒等",
          np.array_equal(dp(zt.tensor(np.ones((2, 3)))).data, np.ones((2, 3))))
    dp.train()
    out_dp = dp(zt.tensor(np.ones((1000,)))).data
    check("Dropout train 期望≈1", abs(out_dp.mean() - 1.0) < 0.08,
          f"mean={out_dp.mean():.3f}")
    # 池化/Flatten/激活
    check("Flatten 形状", Flatten()(zt.tensor(rng.randn(2, 3, 4, 5))).data.shape == (2, 60))
    check("MaxPool2d 形状", MaxPool2d(2)(zt.tensor(rng.randn(1, 1, 6, 6))).data.shape == (1, 1, 3, 3))
    check("AvgPool2d 形状", AvgPool2d(2)(zt.tensor(rng.randn(1, 1, 6, 6))).data.shape == (1, 1, 3, 3))
    check("ReLU 生效", (ReLU()(zt.tensor(np.array([[-1.0, 2.0]]))).data == np.array([[0.0, 2.0]])).all())
    check("GELU 形状", GELU()(zt.tensor(rng.randn(2, 4))).data.shape == (2, 4))

    # ---------- 4. container ----------
    print("\n== 4. 容器 Sequential / ModuleList ==")
    from zerotorch.nn.container import Sequential, ModuleList
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    out = model(zt.tensor(rng.randn(3, 4).astype(np.float64)))
    out.sum().backward()
    check("Sequential 前向+梯度", out.data.shape == (3, 2) and
          all(p.grad is not None for p in model.parameters()))
    check("Sequential 参数命名", [n for n, _ in model.named_parameters()] ==
          ['0.weight', '0.bias', '2.weight', '2.bias'])
    ml = ModuleList([Linear(4, 6), Linear(6, 3)])
    ml.append(Linear(3, 1))
    check("ModuleList 索引/append", len(ml) == 3 and ml[2].out_features == 1)

    # ---------- 5. rnn ----------
    print("\n== 5. RNN / LSTM ==")
    from zerotorch.nn.rnn import RNN, LSTM
    seq_in = rng.randn(2, 4, 3).astype(np.float64)
    rnn = RNN(3, 5)
    for p in rnn.parameters():
        p.data = rng.randn(*p.shape).astype(np.float64) * 0.3
    outs, last_h = rnn(zt.tensor(seq_in))
    check("RNN 输出形状", outs.data.shape == (2, 4, 5) and last_h.data.shape == (4, 5))
    outs.sum().backward()
    check("RNN 梯度有限", all(p.grad is not None and np.isfinite(p.grad).all()
                              for p in rnn.parameters()))
    wp = rnn.w_ih_0
    h1_manual = np.tanh(seq_in[0] @ wp.w.data.T + wp.b.data +
                        np.zeros((4, 5)) @ wp.w_hh.data.T + wp.b_hh.data)
    check("RNN 单步手推", np.allclose(outs.data[0], h1_manual, atol=1e-10))
    lstm = LSTM(3, 5)
    for p in lstm.parameters():
        p.data = rng.randn(*p.shape).astype(np.float64) * 0.2
    outs2, (h, c) = lstm(zt.tensor(seq_in))
    check("LSTM 输出形状", outs2.data.shape == (2, 4, 5) and h.data.shape == (4, 5)
          and c.data.shape == (4, 5))
    outs2.sum().backward()
    check("LSTM 梯度有限", all(p.grad is not None and np.isfinite(p.grad).all()
                               for p in lstm.parameters()))
    lay = lstm.layer0
    gates = seq_in[0] @ lay.w_ih.data.T + lay.b_ih.data + \
            np.zeros((4, 5)) @ lay.w_hh.data.T + lay.b_hh.data
    i_, f_, g_, o_ = np.split(gates, 4, axis=1)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    c1 = sig(f_) * 0 + sig(i_) * np.tanh(g_)
    h1 = sig(o_) * np.tanh(c1)
    check("LSTM 门控手推", np.allclose(outs2.data[0], h1, atol=1e-10))

    # ---------- 6. 端到端：nn.Sequential MLP 训练 ----------
    print("\n== 6. 端到端：nn.Sequential MLP 训练 ==")
    # 注意：必须用可分数据（3 个高斯簇）。随机标签与特征无关，理论最优就是
    # ln(3)≈1.10，模型学不动，测不出收敛能力。
    D = 8
    X = np.concatenate([rng.randn(64, D) + c
                        for c in [(-2.0,) * D, (0.0,) * D, (2.0,) * D]])
    Y = np.repeat([0, 1, 2], 64)
    model = Sequential(Linear(D, 16), ReLU(), Linear(16, 3))
    lr = 0.1
    losses = []
    for _ in range(30):
        loss = zt.F.cross_entropy(model(zt.tensor(X)), Y)
        loss.backward()
        losses.append(float(loss.data))
        for p in model.parameters():
            p.data -= lr * p.grad
            p.grad = None
    check("MLP loss 显著下降", losses[-1] < losses[0] * 0.4,
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"    loss 轨迹: " + " -> ".join(f"{v:.3f}" for v in losses[::7]))

    # ---------- 7. 可选：与 PyTorch 对照 ----------
    print("\n== 7. 与 PyTorch 对照（可选） ==")
    H, C = 16, 3   # 与第 6 节网络结构保持一致
    try:
        import torch
        torch.manual_seed(42)
        # 重新前向收集 zerotorch 梯度
        loss_z = zt.F.cross_entropy(model(zt.tensor(X)), Y)
        loss_z.backward()
        sd_z = {n: p.grad for n, p in model.named_parameters()}
        # PyTorch 同构网络（.double() 统一为 float64，与 zerotorch 权重精度一致）
        mt = torch.nn.Sequential(
            torch.nn.Linear(D, H), torch.nn.ReLU(), torch.nn.Linear(H, C)).double()
        with torch.no_grad():
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), mt.named_parameters()):
                p2.copy_(torch.tensor(p1.data))
        loss_t = torch.nn.functional.cross_entropy(
            mt(torch.tensor(X, dtype=torch.float64)), torch.tensor(Y))
        loss_t.backward()
        diffs = []
        for (n1, _), (n2, p2) in zip(model.named_parameters(), mt.named_parameters()):
            diffs.append(np.abs(p2.grad.numpy() - sd_z[n1]).max())
        check("PyTorch nn.Sequential 梯度对照", max(diffs) < 1e-8,
              f"最大梯度差 {max(diffs):.2e}")
    except ImportError:
        print("  (未安装 torch，跳过对照)")

    print(f"\n{'='*52}")
    print(f"结果: {PASSED} 项通过, {len(FAILURES)} 项失败")
    if FAILURES:
        print("失败项:", FAILURES)
        return 1
    print("L2 nn 层验收通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())

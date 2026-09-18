"""L3 第二批 · 验收测试脚本（data/：Dataset/DataLoader/Sampler/Transforms + 内置数据集）

用法：
    python smoke_test_l3_part2.py        # 与 zerotorch/ 目录同级，或位于其子目录

覆盖：
    1. dataset.py：Dataset 基类 / TensorDataset（对齐、索引、报错）
    2. sampler.py：SequentialSampler / RandomSampler
    3. dataloader.py：batch 形状 / shuffle / drop_last / __len__ / default_collate
    4. transforms.py：ToTensor / Normalize / FlattenImages
    5. datasets/synthetic.py：make_blobs / make_moons / ShapesDetection / CharTextDataset
    6. datasets/mnist.py：类结构 + download=False 报错（真实下载在用户本地执行）
    7. 端到端：DataLoader + make_blobs + Adam 训练 1 个 epoch，loss 下降

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
    data_dir = os.path.join(pkg_dir, 'data')
    required = ['__init__.py', 'dataset.py', 'dataloader.py', 'sampler.py',
                'transforms.py', os.path.join('datasets', '__init__.py'),
                os.path.join('datasets', 'synthetic.py'),
                os.path.join('datasets', 'mnist.py')]
    missing = [f for f in required if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        print(f"  [ERROR] data/ 缺少文件: {missing}")
        return 1
    print("  [PASS] data/ 目录结构完整（8 个必需文件齐）")

    import zerotorch as zt
    from zerotorch.data import (Dataset, TensorDataset, Sampler,
                                 SequentialSampler, RandomSampler,
                                 DataLoader, default_collate,
                                 ToTensor, Normalize, FlattenImages)
    from zerotorch.data.datasets import (make_blobs, make_moons,
                                          ShapesDetection, CharTextDataset)
    from zerotorch.data.datasets.mnist import MNIST
    from zerotorch.nn.container import Sequential
    from zerotorch.nn.layers import Linear, ReLU
    from zerotorch.loss import CrossEntropyLoss
    from zerotorch.optim import Adam

    # ---------- 1. dataset.py ----------
    print("\n== 1. Dataset / TensorDataset ==")
    X = rng.randn(10, 4).astype(np.float32)
    Y = rng.randint(0, 3, 10)
    ds = TensorDataset(X, Y)
    check("TensorDataset 长度", len(ds) == 10)
    x0, y0 = ds[0]
    check("TensorDataset 索引", np.array_equal(x0, X[0]) and y0 == Y[0])
    # 多张量
    Z = rng.randn(10, 2)
    ds3 = TensorDataset(X, Y, Z)
    x3, y3, z3 = ds3[5]
    check("TensorDataset 三张量", np.array_equal(x3, X[5]) and y3 == Y[5]
          and np.array_equal(z3, Z[5]))
    # 长度不一致报错
    try:
        TensorDataset(X, rng.randint(0, 3, 8))
        check("长度不一致报错", False)
    except ValueError:
        check("长度不一致报错", True)
    # 空张量报错
    try:
        TensorDataset()
        check("空张量报错", False)
    except ValueError:
        check("空张量报错", True)
    # 自定义 Dataset 子类
    class MyDS(Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return i * 2
    check("自定义 Dataset", MyDS(5)[3] == 6 and len(MyDS(5)) == 5)

    # ---------- 2. sampler.py ----------
    print("\n== 2. Sampler ==")
    ds10 = TensorDataset(X, Y)
    seq = list(SequentialSampler(ds10))
    check("SequentialSampler 顺序", seq == list(range(10)))
    rnd = list(RandomSampler(ds10))
    check("RandomSampler 是排列", sorted(rnd) == list(range(10)) and len(set(rnd)) == 10)
    check("Sampler 长度", len(SequentialSampler(ds10)) == 10
          and len(RandomSampler(ds10)) == 10)

    # ---------- 3. dataloader.py ----------
    print("\n== 3. DataLoader ==")
    # 顺序加载
    dl = DataLoader(ds10, batch_size=3, shuffle=False)
    batches = list(dl)
    check("DataLoader batch 数（不丢弃）", len(batches) == 4)  # 3,3,3,1
    check("DataLoader __len__", len(dl) == 4)
    bx0, by0 = batches[0]
    check("DataLoader batch 形状", bx0.data.shape == (3, 4) and by0.data.shape == (3,))
    check("DataLoader 输出是 Tensor", isinstance(bx0, zt.Tensor) and isinstance(by0, zt.Tensor))
    # 顺序数据与原始一致
    check("DataLoader 顺序正确", np.array_equal(bx0.data, X[:3])
          and np.array_equal(by0.data, Y[:3]))
    # drop_last
    dl_drop = DataLoader(ds10, batch_size=3, shuffle=False, drop_last=True)
    batches_drop = list(dl_drop)
    check("drop_last 丢弃末尾", len(batches_drop) == 3 and len(dl_drop) == 3)
    # shuffle：所有样本都出现一次
    dl_shuf = DataLoader(ds10, batch_size=3, shuffle=True)
    all_x = np.concatenate([b[0].data for b in dl_shuf], axis=0)
    check("shuffle 覆盖全部样本", sorted(map(tuple, all_x.tolist())) ==
          sorted(map(tuple, X.tolist())))
    # default_collate：标量
    collated = default_collate([1, 2, 3])
    check("default_collate 标量", isinstance(collated, zt.Tensor)
          and collated.data.shape == (3,))
    # default_collate：元组
    collated_t = default_collate([(np.array([1.0, 2.0]), 0),
                                   (np.array([3.0, 4.0]), 1)])
    check("default_collate 元组", isinstance(collated_t, tuple) and len(collated_t) == 2
          and collated_t[0].data.shape == (2, 2))

    # ---------- 4. transforms.py ----------
    print("\n== 4. Transforms ==")
    img = rng.rand(1, 28, 28).astype(np.float32)
    t = ToTensor()(img)
    check("ToTensor", isinstance(t, zt.Tensor) and t.data.shape == (1, 28, 28))
    norm = Normalize(mean=0.5, std=0.5)
    out = norm(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    check("Normalize 数值", np.allclose(out, [-1.0, 0.0, 1.0]), str(out))
    flat = FlattenImages()(rng.rand(2, 1, 4, 4).astype(np.float32))
    check("FlattenImages", flat.data.shape == (2, 16), str(flat.shape))

    # ---------- 5. datasets/synthetic.py ----------
    print("\n== 5. 合成数据集 ==")
    Xb, yb = make_blobs(n_samples=300, n_features=4, centers=3, seed=0)
    check("make_blobs 形状", Xb.shape == (300, 4) and yb.shape == (300,))
    check("make_blobs 标签范围", set(yb.tolist()) == {0, 1, 2})
    between = np.mean([np.linalg.norm(Xb[yb == c].mean(0) - Xb.mean(0)) for c in range(3)])
    within = np.mean([np.mean(np.linalg.norm(Xb[yb == c] - Xb[yb == c].mean(0), axis=1))
                      for c in range(3)])
    check("make_blobs 可分（类内距<类间距）", between > within,
          f"between={between:.2f} within={within:.2f}")
    Xm, ym = make_moons(n_samples=200, noise=0.1, seed=0)
    check("make_moons 形状", Xm.shape == (200, 2) and ym.shape == (200,))

    # ShapesDetection
    sd = ShapesDetection(num_samples=50, img_size=32, seed=0)
    check("ShapesDetection 长度", len(sd) == 50)
    img_s, (boxes_s, labels_s) = sd[0]
    check("ShapesDetection 图像形状", img_s.shape == (1, 32, 32))
    check("ShapesDetection 框形状", boxes_s.shape == (3, 4) and labels_s.shape == (3,))
    check("ShapesDetection 框归一化", boxes_s.min() >= 0 and boxes_s.max() <= 1.0)
    check("ShapesDetection 标签范围", set(labels_s.tolist()).issubset({0, 1, 2}))
    check("ShapesDetection 按 x 排序", all(boxes_s[i, 0] <= boxes_s[i + 1, 0]
                                              for i in range(2)))
    # 图像非空（有像素被绘制）
    check("ShapesDetection 图像非空", img_s.max() > 0)

    # CharTextDataset
    text = "hello world hello zerotorch" * 10
    cds = CharTextDataset(text, seq_len=16, step=8)
    check("CharTextDataset 长度>0", len(cds) > 0)
    cx, cy = cds[0]
    check("CharTextDataset 序列形状", cx.shape == (16,) and cy.shape == (16,))
    check("CharTextDataset target 偏移 1", np.array_equal(cy[:-1], cx[1:]))
    check("CharTextDataset vocab 覆盖", cds.vocab_size == len(set(text)))

    # ---------- 6. datasets/mnist.py ----------
    print("\n== 6. MNIST（结构验证，真实下载在用户本地） ==")
    check("MNIST 类可导入", MNIST is not None)
    # download=False 且无缓存时应报错
    try:
        MNIST(root='/tmp/nonexistent_mnist_dir', train=True, download=False)
        check("MNIST download=False 报错", False)
    except (FileNotFoundError, RuntimeError):
        check("MNIST download=False 报错", True)

    # ---------- 7. 端到端：DataLoader + 训练 ----------
    print("\n== 7. 端到端：DataLoader 喂数据 + Adam 训练 ==")
    Xtr, ytr = make_blobs(n_samples=300, n_features=8, centers=3, seed=1)
    train_ds = TensorDataset(Xtr, ytr)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    net = Sequential(Linear(8, 32), ReLU(), Linear(32, 3))
    loss_fn = CrossEntropyLoss()
    opt = Adam(net.parameters(), lr=1e-2)
    losses = []
    for epoch in range(3):
        for xb, yb in train_dl:
            opt.zero_grad()
            l = loss_fn(net(xb), yb.data)   # yb 是 Tensor，取 data 作为标签
            l.backward()
            opt.step()
        # 全量 loss
        full_loss = float(loss_fn(net(zt.tensor(Xtr)), ytr).data)
        losses.append(full_loss)
    check("DataLoader 训练 loss 下降", losses[-1] < losses[0] * 0.5,
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"    epoch loss: " + " -> ".join(f"{v:.3f}" for v in losses))
    # 准确率
    preds = np.argmax(net(zt.tensor(Xtr)).data, axis=1)
    acc = float(np.mean(preds == ytr))
    check("训练后准确率>80%", acc > 0.8, f"acc={acc:.3f}")

    print(f"\n{'=' * 52}")
    print(f"结果: {PASSED} 项通过, {len(FAILURES)} 项失败")
    if FAILURES:
        print("失败项:", FAILURES)
        return 1
    print("L3 第二批（data/）验收通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())


import random 
import math

class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = data # 真实值
        self.grad = 0.0 # 上游传下来的梯度
        self._backward = lambda: None # 该值的反向梯度
        self._parents = set(_parents) # 哪些参数计算出该值
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # 闭包, 保存现场值
        # 例如 c = a + b; 当c._backward()执行时, a.grad 和 b.grad 的值被更新; 虽然执行到 c._backward()时, a + b 的运算早已经执行完, 但由于闭包机制, 此时依然知道 self->a、other->b
        def _backward(): 
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0 
        
        out._backward = _backward
        return out 

    __radd__ = __add__ 

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data 
            other.grad += out.grad * self.data 
        
        out._backward = _backward
        return out
    
    __rmul__ = __mul__

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data,  (self, other), '-')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * -1.0
        
        out._backward = _backward
        return out 
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += out.grad * (out.data > 0) 
        
        out._backward = _backward
        return out 
    
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad   # tanh 的导数是 1 - tanh^2
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self, ), f'**{k}')

        def _backward():
            self.grad += out.grad * k * (self.data ** (k - 1))
        
        out._backward = _backward
        return out 
    
    def backward(self):
        # 1. 先拓扑排序：从self出发，沿父母方向向上
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build(parent)
                topo.append(v)
        build(self)

        # 2. 输出自己的梯度是 1 
        self.grad = 1.0 

        # 3. 倒着走拓扑序，调用每个节点的 _backward
        for v in reversed(topo):
            v._backward()


class Neuron: # 看图，以创建神经元 h_1 为例
    def __init__(self, n_in=2):
        """
        初始化神经元 n = Neuron(n_in=2)
        """
        # 创建 w_11 和 w_12, 并使用-1～1的随机初始化
        self.w = [Value(random.uniform(-0.5, 0.5)) for _ in range(n_in)]
        # 创建 b , 使用 0 初始化
        self.b = Value(0.0)

    def __call__(self, x):
        """
        输入x 计算神经元值 n(x)
        """
        # act = w_11 * x_1 + w_21 * x_2 + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # h_1 = relu(act) 
        h = act.relu()
        # h = act.tanh()
        return h
        # return act

    def parameters(self):
        # 返回这个神经元所有需要训练的参数 , 即 (w_11, w_21, b)
        return self.w + [self.b]

# random.seed(10) # 固定随机种子÷
# n1 = Neuron(2)
# print([x.data for x in n1.w], n1.b.data, [x.data for x in n1.parameters()])

# x = [1,10]
# print(n1(x).data)

class Layer: # 看图，以隐藏层为例，有3个神经元
    def __init__(self, n_in, n_out):
        # 初始化隐藏层的 3 个神经元；
        # n1=Neuron(2) 其中参数w_11,w_21,b; 
        # n2=Neuron(2) 其中参数w_12,w_22,b;
        # n3类似
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        # 每个神经元算一个输出，返回一个列表, 即 [h_1,h_2,h_3]的值
        # 计算方式为 [n1(x), n2(x), n3(x)]
        return [n(x) for n in self.neurons]

    def parameters(self):
        # 把这一层所有神经元的参数收集起来
        # 即 [w_11,w_21,b,w_12,w_22,b,w_13,w_23,b]
        return [p for n in self.neurons for p in n.parameters()]

# l = Layer(2,3)
# print([x.data for x in l.parameters()])
# print(len(l.parameters()))
# print([d.data for d in l(x)])

class Layer: # 看图，以隐藏层为例，有3个神经元
    def __init__(self, n_in, n_out):
        # 初始化隐藏层的 3 个神经元；
        # n1=Neuron(2) 其中参数w_11,w_21,b; 
        # n2=Neuron(2) 其中参数w_12,w_22,b;
        # n3类似
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        # 每个神经元算一个输出，返回一个列表, 即 [h_1,h_2,h_3]的值
        # 计算方式为 [n1(x), n2(x), n3(x)]
        return [n(x) for n in self.neurons]

    def parameters(self):
        # 把这一层所有神经元的参数收集起来
        # 即 [w_11,w_21,b,w_12,w_22,b,w_13,w_23,b]
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_in, outs):
        # outs 是每层神经元数量，比如 [4, 4, 1] 表示两个隐藏层各 4 个，输出层 1 个
        sz = [n_in] + outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(outs))]

    def __call__(self, x):
        # 前向：一层一层往后传
        for layer in self.layers:
            x = layer(x)
        return x[0]  # 最后一层只有 1 个神经元，直接返回

    def parameters(self):
        # 收集整个网络所有参数
        return [p for layer in self.layers for p in layer.parameters()]

model = MLP(2, [4, 4, 1])   # 2 输入，1 个隐藏层 3 个神经元，1 输出
data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]

for k in range(200):
    # 前向：算 loss（这里用最简单的 (pred - y)^2）
    total = Value(0.0)
    for x, y in data:
        pred = model(x)
        total = total + (pred - Value(float(y))) ** 2

    # 清零梯度（注意：必须每步清零，否则会累加）
    for p in model.parameters():
        p.grad = 0.0
    # 反向
    total.backward()
    # 梯度下降：沿着梯度反方向走一小步
    for p in model.parameters():
        p.data -= 0.05 * p.grad

    if k % 20 == 0:
        print(f"step {k:3d}  loss = {total.data:.4f}")


for x, y in [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0),([1.5,-0.5],1), ([-0.5,1.5],1), ([-0.5,-0.5],0),([0.5,0.5],0)]:
    pred = model(x)
    print(f"输入 {x}，预测 {pred.data:.3f}，标签 {y}")

import numpy as np
import matplotlib.pyplot as plt

# 在 [-0.5, 1.5] × [-0.5, 1.5] 上撒 100×100 的网格
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                     np.linspace(-0.5, 1.5, 100))
zz = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        zz[i, j] = model([xx[i, j], yy[i, j]]).data

plt.figure(figsize=(5, 5))
# 决策边界：用颜色表示模型输出
plt.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.6)
plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
# 画出 4 个训练点
for x, y in data:
    plt.plot(x[0], x[1], "o", color="red" if y else "blue",
             markersize=12, markeredgecolor="black")
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("XOR 决策边界")
plt.savefig("xor_boundary.png", dpi=100, bbox_inches="tight")
plt.show()
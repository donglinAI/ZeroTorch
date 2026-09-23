
import numpy as np

class Tensor:
    
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data) # 和self.data形状一致，因为给每个data都配grad
        self._backward = lambda: None # 当前张量调用该方法，可赋值父节点梯度
        self._parents = set() # 父节点集合，不分先后顺序
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add.apply(self, other)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return MatMul.apply(self, other)

    def backward(self):
        # 拓扑排序
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build(p)
        build(self)

        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

class Function:
    @staticmethod
    def forward(ctx, *inputs):
        """前向计算。

        参数：
            ctx: 上下文对象，前向往里存中间结果，反向再取出来
            inputs: 前向输入的数据（纯数组，不是 Tensor ）
        返回：
            前向结果（纯数组） 
        """
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, out_grad):
        """反向计算。

        参数：
            ctx: 上下文对象，取出 forward 存下的中间结果
            out_grad: 上游传来的“输出对我”的梯度，形状和输出一致
        返回：
            各输入的梯度（元组，顺序和 inputs 对应）
        """
        raise NotImplementedError
    
    @classmethod 
    def apply(cls, *inputs): # 直接能构建一个Function类？
        """算子的统一入口：跑前向、记父母、登记反向。

        参数：
            cls: 具体算子类（Add / MatMul / ...）; 之前在 Value 类中通过 闭包机制 实现，现在抽象成单独的类
            inputs: 前向输入的 Tensor
        返回：
            输出 Tensor （挂着 _parant 和 _backward）
        """
        # 1. 前向计算
        ctx = type("Ctx", (), {})()
        out_data = cls.forward(ctx, *[i.data for i in inputs])
        out = Tensor(out_data)

        # 2. 记录父母（为拓扑排序准备）
        out._parents = set(inputs)

        # 3. 登记 _backward: 反向调用本算子的 backward ，把梯度累加回输入
        def _backward():
            input_grads = cls.backward(ctx, out.grad) # out.grad 是矩阵形式么？dy？input_grads是什么元组 还是 矩阵
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads, )
            for inp, g in zip(inputs, input_grads):
                inp.grad += g  # inp 和 g 应该还是矩阵
        out._backward = _backward

        return out 
        

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y
    
    @staticmethod
    def backward(ctx, grad):
        return grad, grad

class MatMul(Function):
    @staticmethod
    def forward(ctx, X, W):
        ctx.saved = (X, W)
        return X @ W 

    @staticmethod
    def backward(ctx, grad):
        X, W = ctx.saved
        dx = grad @ W.T
        dw = X.T @ grad 
        return dx, dw


x_data = np.array(
    [[0, 0],
    [1, 1],
    [0, 1],
    [1, 0]], dtype=np.float64
)
x_data = Tensor(x_data)

y_data = [0,0,1,1]
y_data = Tensor(y_data)

W1 = Tensor(np.random.uniform(-1,1,(2,3))) # numpy 和 Python 的一些方法不同
b1 = Tensor(np.zeros_like(3,))

W2 = Tensor(np.random.uniform(-1,1,(3,1)))
b2 = Tensor(np.zeros_like(1,))

print(W1.data,b1.data,W2.data,b2.data)

print(x_data @ W1)
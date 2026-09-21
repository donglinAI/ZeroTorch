


class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = data # 真实值
        self.grad = 0.0 # 上游传下来的梯度
        self._backward = lambda: None # 该值的反向梯度
        self._parents = set(_parents) # 哪些参数计算出该值
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

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

if __name__ == "__main__":

    a = Value(2.0)
    b = Value(-3.0)
    c = a + b # -1.0
    d = a * b # -6.0
    e = c * d # 6.0
    e.backward()

    print("a的梯度:", a.grad)
    print("b的梯度:", b.grad)
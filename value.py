import math

class Value:

    def __init__(self, data, _children = (), _op = '', label = ''):
        
        self.data = data 
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        
        other =  other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out 
    
    def __pow__(self, other):

        assert isinstance(other, (int, float)), "onlyu supports int/float"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')
        def _backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        return other + (-self)
    
    def __rpow__(self, other):
        return other ** self
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * self**-1 
    
    def __neg__(self):
        return self * (-1)
    
    def exp(self):
      x = self.data
      out =  Value(math.exp(x), (self, ), 'exp')
      def _backward():
          self.grad += out.data * out.grad
      out._backward = _backward
      return out 
    
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward= _backward 

        return out
    
    def sigmoid(self):
        
        x = self.data
        y = 1/(1 + math.exp(-x))
        out = Value(y, (self, ), 'sigmoid')
        def _backward(): 
            self.grad += y * (1 - y) * out.grad
        out._backward = _backward
        return out 

    def tanh(self):
        
        x = self.data
        y = (math.exp(self.data) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(y, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - y ** 2) * out.grad
        out._backward = _backward
        return out 

    def ReLU(self):
        
        out = Value(self.data if self.data >= 0 else 0, (self, ), 'ReLU')
        def _backward():
            self.grad += (0 if self.data <= 0 else 1) * out.grad
        out._backward = _backward
        return out

    def Leaky_ReLU(self, decay_rate = 0.1):

        out = Value(decay_rate * self.data if self.data <= 0 else self.data, (self, ), 'Leaky ReLU')
        def _backward():
            self.grad += (decay_rate if self.data <=0 else 1) * out.grad
        out._backward = _backward
        return out 
    
    def ELU(self, decay_rate = 0.1):
        
        x = decay_rate * (math.exp(self.data) - 1)
        out = Value(self.data if self.data >= 0 else x, (self, ), 'ELU')
        def _backward():
            self.grad += (1 if self.data >= 0 else decay_rate * math.exp(self.data)) * out.grad
        out._backward = _backward
        return out 

    def backward(self):
        topos = []
        visited = set()
        def build_topo(v):
            visited.add(v)
            for child in v._prev:
                if child not in visited:
                    build_topo(child)
            topos.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topos):
            node._backward()


import random 
import numpy as np 
import math
import matplotlib.pyplot as plt
from Graph import draw_dot

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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')
        def _backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
            other.grad += (math.log(self.data) * out.data) * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        return self - other
    
    def __rpow__(self, other):
        return self ** other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __neg__(self):
        return self * (-1)
    
    def exp(self):
      x = self.data
      out =  Value(math.exp(x), (self, ), 'exp')
      def _backward():
          self.grad += out.data * out.grad
      out._backward = _backward
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

    



#def main():
#    a = Value(3)    ; a.label = 'a'
#    b = Value(4)    ; b.label = 'b'
#    c = a * b       ; c.label = 'c'
#    d = c.sigmoid() ; d.label = 'd'
#    e = d.tanh()    ; e.label = 'e' 
#    f = e.ReLU()    ; f.label = 'f'
#    g = f.Leaky_ReLU() ; g.label = 'g'
#    h = g.ELU()     ; h.label = 'h'
#
#    h.backward()
#    draw_dot(h)
#main()



















































'''
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

b = Value(6.88813735870195432, label='b')

x1w1 = x1 * w1; x1w1.label = 'x1w1'
x2w2 = x2 * w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b; n.label = 'n'
e = (2*n).exp() ; e.label = 'e'
o = (e - 1) / (e + 1); o.label = 'o'
o.backward()
draw_dot(o)
w =  [Value(random.uniform(-1,1)) for _ in range(3)]
b =  Value(random.uniform(-1,1))

print(b)
x = [2,2,2]
a = sum([w * i for w,i in zip(w, x)],b)
    
print (a)
'''
import random
from value import Value
from loss import *


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neurons(Module):

    def __init__(self, nin, activation ='ReLU'):
        self.w = [Value(random.uniform(-1,1))  for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self._act = activation
    def __call__(self, x):  
        non_linear = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self._act == 'ReLU':
            out = non_linear.ReLU()
        elif self._act == 'tanh':
            out = non_linear.tanh()
        elif self._act == 'sigmoid':
            out = non_linear.sigmoid()
        elif self._act == 'leaky_ReLU':
            out = non_linear.Leaky_ReLU()
        elif self._act == 'ELU':
            out = non_linear.ELU()
        else:
            raise ValueError(f"Unsupported activation: {self._act}")
        return out 
    
    def parameters(self):
        return self.w + [self.b]
    
class Layers(Module):

    def __init__(self, nin, nout, activation = 'ReLU'):
        self.neurons = [Neurons(nin, activation) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs 
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
     
class MLP(Module):

    def __init__(self, nin, nout, activation = 'ReLU'):
        sz = [nin] + nout
        self.layers = [Layers(sz[i],sz[i+1], activation) for i in range(len(nout))]
 
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]    
    



    




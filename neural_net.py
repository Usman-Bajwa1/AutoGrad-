import numpy as np
import random
import math
from value import Value
from Graph import draw_dot

class Neurons:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1))  for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):  
        out =  sum((w * i for w, i in zip(self.w, x)), self.b)
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layers:

    def __init__(self, nin, nout):
        self.neurons = [Neurons(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs 
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
     
class MLP:

    def __init__(self, nin, nout):
        sz = [nin] + nout
        self.layers = [Layers(sz[i],sz[i+1]) for i in range(len(nout))]
 
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]    
    
class Loss:

    def __init__(self):
        pass
    def loss(self, y_pred, y):
        out = sum([(yout - ygt) ** 2 for ygt,yout in zip(y, y_pred)])
        return out 

xs = [
    [2.0, 3.0 ,-1.0],
    [3.0, -1.0 ,0.5],
    [8.5, 1.0 ,1.0],
    [1.0, 1.0 ,-1.0]
]

ys = [1.0, -1.0, -1.0, 1.0] 

n = MLP(3, [4,4,1])

ypred = [n(x) for  x in xs]
print(ypred)
#loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

los = Loss()
loss = los.loss(ypred, ys)
loss.backward()
print(n.layers[2].neurons[0].w[3].grad) 
print(len(n.parameters()))

for p in n.paramters():
    p.data += -0.01 + p.grad



    




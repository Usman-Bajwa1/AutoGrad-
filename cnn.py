import random
from neural_net import Module
import numpy as np
from value import Value

class Conv2d(Module):

    def __init__(self, shape = (6,3,3,3), pad = 2, stride = 1):
        self.shape = shape
        self.out_channel, self.F, self.HH, self.WW = shape
        self.filter = Value(np.random.rand(self.out_channel, self.F, self.HH, self.WW))
        self.pad = pad
        self.stride = stride    
    def __call__(self,x):        
        N, in_channel, H, W = x.shape
        out_channel, F, HH, WW = self.shape
        pad = self.pad
        stride = self.stride
        x_pad = np.pad(x, ((0,0),(0,0),(pad, pad),(pad, pad)), 'constant')
        
        out_h = 1 + (H + 2*pad - HH) // stride
        out_w = 1 + (W + 2*pad - WW) // stride     
        out = np.zeros((N, out_channel, out_h, out_w))

        for bs in range(N):
            for oc in range(out_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        region_sum = Value(0)
                        for ic in range(in_channel):
                            region = x_pad[bs, ic, i*stride: i*stride + HH, j*stride: j*stride + WW]
                            region_sum += np.sum(region * self.filter[oc, ic], start = Value(0))
                        out[bs,oc,i,j] = region_sum 
        
        



def main():

    x = np.random.randn(4, 3, 5, 5)
    n = Conv2d()
    a = n(x)
    print(a)

main()
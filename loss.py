import numpy as np
#from neural_net import MLP
from value import Value


def RMSE_loss(y_hat, y):
    if not isinstance(y_hat, list):
        y_hat = [y_hat]
    if not isinstance(y, list):
        y = [y]
    out = sum([(yout - ygt) ** 2 for ygt,yout in zip(y_hat, y)], start = Value(0))
    out /= len(y)
    out = out ** 0.5 
    return out

def main():
    pass        
main()
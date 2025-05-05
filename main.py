from neural_net import *
from loss import *
from value import * 
import random 

def main():
    X = [[random.uniform(1,3) for _ in range(10)],
         [random.uniform(1,5) for _ in range(10)],
         [random.uniform(1,6) for _ in range(10)],
         [random.uniform(1,4) for _ in range(10)],
         [random.uniform(1,2) for _ in range(10)],
         [random.uniform(1,7) for _ in range(10)],
         [random.uniform(1,9) for _ in range(10)],
         [random.uniform(1,10) for _ in range(10)]
        ]
    y = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0]
    
    model = MLP(10, [16,16,1], activation='tanh')
    for k in range(10):

        y_pred = [model(x) for x in X]
        loss = RMSE_loss(y_pred, y)

        model.zero_grad()

        loss.backward()

        for p in model.parameters():
            p.data -= 0.01 + p.grad

        print(f"The loss after {k} iteration is {loss.data: .4f}")

main() 
        

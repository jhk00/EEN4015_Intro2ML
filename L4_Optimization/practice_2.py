import torch

size = 100
epochs = 1000
start = -10
end = 10
class SGD:
    def __init__(self, lr):
        self.lr = lr
    
    def update(self, params, grads):
        with torch.no_grad():
            params -= self.lr * grads


a3 = 4.0
a2 = 3.0
a1 = 2.0
a0 = 1.0

def poly(x):
    return a3*x**3 + a2*x**2 + a1*x + a0

X = torch.arrange(start,end,(start-end)/size,dtype=torch.float32)
Y = poly(X) + torch.normal(0,0.75,size=(size,))

opt = SGD(lr=0.01)

for epoch in range(epochs):

    index = torch.randperm(n)
    index_sampled = index[:m]

    x_batch = x[index_sampled, :] # (100, 100)
    y_batch = Y[index_sampled, :] # (100, 10)




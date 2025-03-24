import torch 

import torch.nn as nn 

import torch.optim as optim 

d = 4 

n = 1000 

noise = 1e-2 

num_iter = 10000 

batch_size = 10 

lr = 1e-2 

# Generate random data 

t = (torch.rand(n) * 2 - 1) * 3.14159 

a_true = torch.randn(d) 

y = torch.zeros(n) 

for i in range(d): 

    y += a_true[i] * torch.cos(i * t) 

y += torch.randn(n) * noise 

 

# build model 

class CosSeries(nn.Module): 

    def __init__(self, d): 

        super(CosSeries, self).__init__() 

        self.d = d 

        self.a = nn.Parameter(torch.randn(d)) 

    def forward(self, t): 

        y = torch.zeros(t.shape) 

        for i in range(self.d): 

            y += self.a[i] * torch.cos(i * t) 

        return y 

model = CosSeries(d) 

optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9) 

criterion = nn.MSELoss() 

for i in range(num_iter): 

    idx = torch.randperm(n)[:batch_size] 

    t_batch = t[idx] 

    y_pred = model(t_batch) 

    loss = criterion(y_pred, y[idx]) 

    loss.backward() 

    optimizer.step() 

    optimizer.zero_grad() 

    print(f'Iter {i}: loss={loss.item()}') 
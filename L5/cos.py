import torch

d = 4
n = 10000
noise = 1e-2

num_iter = 1000

# Generate random data
t = (torch.rand(n) * 2 - 1) * 3.14159
a_true = torch.randn(d)
y = torch.zeros(n)
for i in range(d):
    y += a_true[i] * torch.cos(i*t)
y += torch.randn(n) * noise

a_model = torch.randn(d, requires_grad=True)

for i in range(num_iter):
    idx = torch.randperm(n)[:batch_size]
    t_batch = t[idx]
    
    y
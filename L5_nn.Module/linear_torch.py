import torch

# numbers
d_input = 100 # dimension of input
d_output = 10 # dimension of output
n = 1000 # number of samples
noise = 0.01 # noise level
m = 100 # size of minibatch
epochs = 1000


# randomly generated data
x = torch.randn(n, d_input) # (1000, 100) 
W_true = torch.randn(d_input, d_output) # (100, 10)
Y = x @ W_true # ground truth 
# (1000,10)
Y = Y + torch.randn(n, d_output) * noise # adding noise

# randomly initialized model
W_model = torch.randn(d_input, d_output, requires_grad = True)
# (100, 10)

""" 
SGD 구현 , linear regression 푸는 알고리즘 만들기
torch의 autograd 기능 이용해 직접 구현하기
W의 추정값을 SGD를 이용한 학습을 통해 찾기

W의 정답과 추정값이 얼마나 근접한지 오
"""



class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression,self).__init__()
        self.linear = torch.nn.Linear(input_dim,output_dim,bias=False)
        # (100,10)
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = LinearRegression(d_input,d_output)           
criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

for epoch in range(epochs):


    index = torch.randperm(n)
    index_sampled = index[:m]

    x_batch = x[index_sampled, :] # (100, 100)
    y_batch = Y[index_sampled, :] # (100, 10)
    # 그냥 m-1 숫자 넣으면 벡터로 아웃풋 나오고
    # 인덱싱에 텐서 넣어야 텐서가 아웃풋으로 나옴
    # 그래서 index_sampled 사용

    y_pred = model(x_batch)

    loss = criterion(y_pred,y_batch)
    opt.zero_grad()
    loss.backward()

    opt.step()

    if (epoch + 1) % 10 ==0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")


diff = torch.norm(W_true - W_model).item()
print(f"{diff:.6f}")

    


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

W의 정답과 추정값이 얼마나 근접한지 오차 확인하기
"""
class SGD:
    def __init__(self, lr):
        self.lr = lr
    
    def update(self, params, grads):
        with torch.no_grad():
            params -= self.lr * grads
        # for key in params.keys():
        #     params[key] = params[key] - self.lr*grads[key]
            # params['W'] = W_model
            # grads['W'] = W_model.grad
            # in-place error
            
opt = SGD(lr=0.01)

for epoch in range(epochs):


    index = torch.randperm(n)
    index_sampled = index[:m]

    x_batch = x[index_sampled, :] # (100, 100)
    y_batch = Y[index_sampled, :] # (100, 10)
    # 그냥 m-1 숫자 넣으면 벡터로 아웃풋 나오고
    # 인덱싱에 텐서 넣어야 텐서가 아웃풋으로 나옴
    # 그래서 index_sampled 사용

    y_pred = x_batch @ W_model # (100, 10)

    loss = ((y_pred - y_batch) ** 2).mean()
    # MSE loss : Linear regression에서 자주 쓰임

    loss.backward()

    opt.update(W_model, W_model.grad)

    W_model.grad.zero_()

    if (epoch + 1) % 10 ==0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")


diff = torch.norm(W_true - W_model).item()
print(f"{diff:.6f}")

    


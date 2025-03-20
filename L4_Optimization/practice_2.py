import torch

size = 100
epochs = 1000
start = -10
end = 10
n = 1000

class SGD:
    def __init__(self, lr):
        self.lr = lr
    
    def update(self, params, grads):
        with torch.no_grad():
            params -= self.lr * grads

# 실제 함수 계수 (정답)
a3 = 4.0
a2 = 3.0
a1 = 2.0
a0 = 1.0

def poly(x):
    return a3 * x**3 + a2 * x**2 + a1 * x + a0

# 데이터셋 생성
X = torch.linspace(start, end, steps=n)  # shape: (1000,)
Y = poly(X) + torch.normal(0, 0.75, size=(n,))

opt = SGD(lr=1e-6)

# 학습할 파라미터 초기화 (requires_grad 오타 수정)
a3_pred = torch.randn(1, requires_grad=True)
a2_pred = torch.randn(1, requires_grad=True)
a1_pred = torch.randn(1, requires_grad=True)
a0_pred = torch.randn(1, requires_grad=True)

for epoch in range(epochs):

    index = torch.randperm(n)
    index_sampled = index[:size]
    
    x_batch = X[index_sampled]
    y_batch = Y[index_sampled]

    # 예측값 계산
    y_pred = a3_pred * x_batch**3 + a2_pred * x_batch**2 + a1_pred * x_batch + a0_pred

    # 손실 계산 (MSE)
    loss = ((y_pred - y_batch) ** 2).mean()
    
    # 역전파: 파라미터들의 그라디언트 계산
    loss.backward()
    
    # 파라미터 업데이트
    opt.update(a3_pred, a3_pred.grad)
    opt.update(a2_pred, a2_pred.grad)
    opt.update(a1_pred, a1_pred.grad)
    opt.update(a0_pred, a0_pred.grad)
    
    # 그라디언트 초기화 (누적 방지)
    a3_pred.grad.zero_()
    a2_pred.grad.zero_()
    a1_pred.grad.zero_()
    a0_pred.grad.zero_()
    
    # 100 epoch마다 학습 상황 출력
    if (epoch+1) % 100 == 0:
        print(f"[Epoch {epoch+1}/{epochs}] loss = {loss.item():.4f}")
        print(f"  a3_pred = {a3_pred.item():.4f}, "
              f"a2_pred = {a2_pred.item():.4f}, "
              f"a1_pred = {a1_pred.item():.4f}, "
              f"a0_pred = {a0_pred.item():.4f}")

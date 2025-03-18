import torch
import matplotlib.pyplot as plt 

def g(x):
    a = torch.pow(x * 2, 1/3) * torch.pow(x + 5, 1/2)
    b = torch.pow(4 * x - 1, 9)
    return a / b
 
# 입력값과 requires_grad 설정
x = torch.tensor([3.0], requires_grad=True)

# 함수 g(x) 계산
y = g(x)

# 역전파로 도함수 계산
y.backward()

# 출력
print(f"g({x.item()}) = {y.item()}")       # g(x)의 함수값
print(f"g'({x.item()}) = {x.grad.item()}") # g(x)의 도함수값

# item() 메서드란?
# 텐서 내부에 하나의 값만 포함된 경우, 이 값을 파이썬 스칼라 값으로 변환



x_values = torch.linspace(0.6, 10.0, 300)  # 0.5가 아닌 0.6로 살짝 올림 (안정성)
g_values = []
grad_values = []

for x_val in x_values:
    x = torch.tensor([x_val.item()], requires_grad=True)
    y = g(x)
    y.backward()

    g_values.append(y.item())
    grad_values.append(x.grad.item())

# ----- 서로 다른 y축으로 그래프 그리기 -----
fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'tab:blue'
ax1.set_xlabel('x')
ax1.set_ylabel('g(x)', color=color1)
ax1.plot(x_values.tolist(), g_values, color=color1, label='g(x)', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# 두 번째 축
ax2 = ax1.twinx()  
color2 = 'tab:orange'
ax2.set_ylabel("g'(x)", color=color2)
ax2.plot(x_values.tolist(), grad_values, color=color2, label="g'(x)", linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title("g(x) and g'(x) with Dual Y-Axis")
fig.tight_layout()
plt.grid(True)
plt.show()
import torch
import matplotlib.pyplot as plt
import numpy as np

def f(a, b, x1, x2):
    z = a * x1 + b * x2
    num = torch.tanh(z)
    denom = 1.0 + torch.exp(-torch.cos(z))
    return num / denom

"""
 (a, b, x1, x2)에 대해 f와 편미분값을 구하는 함수
    - df/da, df/db, df/dx1, df/dx2
"""
def compute_gradients(a_val, b_val, x1_val, x2_val):
    a = torch.tensor(a_val, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(b_val, requires_grad=True, dtype=torch.float32)
    x1 = torch.tensor(x1_val, requires_grad=True, dtype=torch.float32)
    x2 = torch.tensor(x2_val, requires_grad=True, dtype=torch.float32)

    # f 계산
    y = f(a, b, x1, x2)
    
    y.backward()
    
    y_val = y.item()
    da = a.grad.item()
    db = b.grad.item()
    dx1 = x1.grad.item()
    dx2 = x2.grad.item()
    
    return y_val, da, db, dx1, dx2

def main():
    a_val = 1.0
    b_val = 2.0
    x1_val = 3.0
    x2_val = 4.0

    y_val, da, db, dx1, dx2 = compute_gradients(a_val, b_val, x1_val, x2_val)

    # 출력 먼저!
    print(f"f(a={a_val}, b={b_val}, x1={x1_val}, x2={x2_val}) = {y_val}")
    print(f"df/da = {da}")
    print(f"df/db = {db}")
    print(f"df/dx1 = {dx1}")
    print(f"df/dx2 = {dx2}")

    # 이후에 그래프 시각화 진행
    # 첫 번째 시각화
    a_fixed = 1.0
    b_fixed = 2.0

    x1_vals = torch.linspace(-2.0, 2.0, steps=20)
    x2_vals = torch.linspace(-2.0, 2.0, steps=20)

    X1, X2 = np.meshgrid(x1_vals.numpy(), x2_vals.numpy())
    U = np.zeros_like(X1)
    V = np.zeros_like(X2)

    for i in range(len(x1_vals)):
        for j in range(len(x2_vals)):
            _, _, _, dfdx1, dfdx2 = compute_gradients(a_fixed, b_fixed, X1[i, j], X2[i, j])
            U[i, j] = dfdx1
            V[i, j] = dfdx2

    plt.figure(figsize=(8, 6))
    plt.quiver(X1, X2, U, V, color='blue')
    plt.title("Gradient Field wrt (x1, x2) (a=1.0, b=2.0)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

    # 두 번째 시각화
    x1_fixed = 1.0
    x2_fixed = -1.0

    a_vals = torch.linspace(-2.0, 2.0, steps=20)
    b_vals = torch.linspace(-2.0, 2.0, steps=20)

    A, B = np.meshgrid(a_vals.numpy(), b_vals.numpy())
    U_ab = np.zeros_like(A)
    V_ab = np.zeros_like(B)

    for i in range(len(a_vals)):
        for j in range(len(b_vals)):
            _, da, db, _, _ = compute_gradients(A[i, j], B[i, j], x1_fixed, x2_fixed)
            U_ab[i, j] = da
            V_ab[i, j] = db

    plt.figure(figsize=(8, 6))
    plt.quiver(A, B, U_ab, V_ab, color='red')
    plt.title("Gradient Field wrt (a, b) (x1=1.0, x2=-1.0)")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.grid(True)
    plt.show()

# 실제 실행 시에는 아래 main() 호출
if __name__ == "__main__":
    main()

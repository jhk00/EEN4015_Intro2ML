import torch
import matplotlib.pyplot as plt
import numpy as np

def f(a, b, x1, x2):
    # a, b, x1, x2 모두 스칼라 텐서라고 가정
    z = a * x1 + b * x2
    num = torch.tanh(z)
    denom = 1.0 + torch.exp(-torch.cos(z))
    return num / denom

#----------------------------------------------------------
# 2) 특정 (a, b, x1, x2)에 대해 f와 편미분값을 구하는 함수
#    - df/da, df/db, df/dx1, df/dx2
#----------------------------------------------------------
def compute_gradients(a_val, b_val, x1_val, x2_val):
    # requires_grad=True 설정
    a = torch.tensor(a_val, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(b_val, requires_grad=True, dtype=torch.float32)
    x1 = torch.tensor(x1_val, requires_grad=True, dtype=torch.float32)
    x2 = torch.tensor(x2_val, requires_grad=True, dtype=torch.float32)

    # f 계산
    y = f(a, b, x1, x2)
    
    # 역전파로 모든 편미분 계산
    y.backward()
    
    # .item()을 통해 스칼라값 추출
    y_val = y.item()
    da = a.grad.item()
    db = b.grad.item()
    dx1 = x1.grad.item()
    dx2 = x2.grad.item()
    
    return y_val, da, db, dx1, dx2

#----------------------------------------------------------
# 3) main: 문제 상황에 맞춰 단계별로 구현
#----------------------------------------------------------
def main():
    #-------------------------
    # 3-1) (x1, x2)에 대한 gradient field 시각화
    #      a, b는 고정
    #-------------------------
    a_fixed = 1.0
    b_fixed = 2.0

    # x1, x2 범위를 -2 ~ 2로 설정(임의)
    x1_vals = torch.linspace(-2.0, 2.0, steps=20)
    x2_vals = torch.linspace(-2.0, 2.0, steps=20)

    # meshgrid 생성 (numpy 배열로 변환)
    X1, X2 = np.meshgrid(x1_vals.numpy(), x2_vals.numpy())

    # U, V는 각각 df/dx1, df/dx2
    U = np.zeros_like(X1)
    V = np.zeros_like(X2)

    # 격자 각 지점에서 편미분 계산
    for i in range(len(x1_vals)):
        for j in range(len(x2_vals)):
            _, _, _, dfdx1, dfdx2 = compute_gradients(a_fixed, b_fixed, X1[i, j], X2[i, j])
            U[i, j] = dfdx1
            V[i, j] = dfdx2

    # quiver로 벡터 필드 시각화
    plt.figure(figsize=(8, 6))
    plt.quiver(X1, X2, U, V, color='blue')
    plt.title("Gradient Field wrt (x1, x2) (a=1.0, b=2.0)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

    #-------------------------
    # 3-2) (a, b)에 대한 gradient field 시각화
    #      x1, x2는 고정
    #-------------------------
    x1_fixed = 1.0
    x2_fixed = -1.0

    # a, b 범위를 -2 ~ 2로 설정(임의)
    a_vals = torch.linspace(-2.0, 2.0, steps=20)
    b_vals = torch.linspace(-2.0, 2.0, steps=20)

    # meshgrid 생성
    A, B = np.meshgrid(a_vals.numpy(), b_vals.numpy())

    # U_ab, V_ab는 각각 df/da, df/db
    U_ab = np.zeros_like(A)
    V_ab = np.zeros_like(B)

    # 격자 각 지점에서 편미분 계산
    for i in range(len(a_vals)):
        for j in range(len(b_vals)):
            _, da, db, _, _ = compute_gradients(A[i, j], B[i, j], x1_fixed, x2_fixed)
            U_ab[i, j] = da
            V_ab[i, j] = db

    # quiver로 벡터 필드 시각화
    plt.figure(figsize=(8, 6))
    plt.quiver(A, B, U_ab, V_ab, color='red')
    plt.title("Gradient Field wrt (a, b) (x1=1.0, x2=-1.0)")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.grid(True)
    plt.show()

    #-------------------------
    # 3-3) 특정 (a, b, x1, x2)에 대해 모든 편미분값 동시에 구하기
    #-------------------------
    a_val = 1.0
    b_val = 2.0
    x1_val = 3.0
    x2_val = 4.0

    y_val, da, db, dx1, dx2 = compute_gradients(a_val, b_val, x1_val, x2_val)

    print(f"f(a={a_val}, b={b_val}, x1={x1_val}, x2={x2_val}) = {y_val}")
    print(f"df/da = {da}")
    print(f"df/db = {db}")
    print(f"df/dx1 = {dx1}")
    print(f"df/dx2 = {dx2}")

# 실제 실행 시에는 아래 main() 호출
if __name__ == "__main__":
    main()

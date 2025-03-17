import torch 

# 정수형 텐서 a 생성 (기본적으로 torch.int64)
a = torch.tensor([1, 2, 3])  
# a = tensor([1, 2, 3])  --> [1, 2, 3]이 저장된 1차원 텐서

# 아래 두 줄은 주석 처리되어 실행되지 않음
# a = torch.tensor([1., 2., 3.])  
# a = torch.tensor([1, 2, 3], dtype=torch.float32)  

print(a)  
# 출력 예: tensor([1, 2, 3])
# → a가 [1, 2, 3] 값을 가진 텐서임을 출력함

print(a.dtype)  
# 출력 예: torch.int64
# → a가 정수형(int64)로 생성되었기 때문임

print(a.shape)  
# 출력 예: torch.Size([3])
# → a의 차원이 1차원이고, 길이가 3임을 나타냄

# 2차원 텐서 b 생성
b = torch.tensor([[1, 2, 3], [4, 5, 6]])  
# b = [[1,2,3],
#      [4,5,6]]

print(b)  
# 출력 예: tensor([[1, 2, 3],
#                  [4, 5, 6]])
# → b의 내용을 2행 3열의 행렬로 출력

print(b.shape)  
# 출력 예: torch.Size([2, 3])
# → b의 shape가 (2, 3)임을 보여줌

# c에 여러 텐서 생성 함수 적용 (마지막으로 할당된 값만 남음)
c = torch.zeros(3, 4)   # 3x4 제로 텐서 생성 (이후 덮어씌워짐)
c = torch.ones(3, 4)    # 3x4 원소가 모두 1인 텐서 생성
c = torch.rand(3, 4)    # 3x4 균등분포(0~1) 난수 텐서 생성
c = torch.randn(3, 4)   # 3x4 정규분포 난수 텐서 생성
c = torch.eye(3)        # 3x3 단위 행렬 생성 (대각선이 1)
c = torch.arange(10)    # 0부터 9까지의 정수로 1차원 텐서 생성

print(c)  
# 출력 예: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# → c는 마지막에 torch.arange(10)로 재할당되어 1차원 텐서가 됨

# d에 여러 텐서 생성 함수 적용 (최종 d는 스칼라 1.0)
d = torch.zeros(3)        # shape: (3,)
d = torch.zeros(3, 4)     # shape: (3, 4)
d = torch.zeros(3, 4, 5)   # shape: (3, 4, 5)
d = torch.zeros([])       # 0차원 텐서 (스칼라) 0을 생성
d = torch.tensor(1.)      # 스칼라 텐서 1.0으로 재할당

print(d)  
# 출력 예: tensor(1.)
# → d는 최종적으로 1.0을 저장한 스칼라 텐서임

# x와 y에 2x2 난수 텐서를 생성한 후, 다양한 연산을 수행 (최종 z는 행렬곱 결과)
x = torch.randn(2, 2)   # 예: tensor([[a, b], [c, d]])
y = torch.randn(2, 2)   # 예: tensor([[e, f], [g, h]])

z = x + y   # 덧셈 → 임시 결과
z = x - y   # 뺄셈 → 임시 결과
z = x * y   # 원소별 곱셈 → 임시 결과
z = x / y   # 원소별 나눗셈 → 임시 결과
z = x @ y   # 행렬 곱셈 (최종 결과; dot product 연산)

print(x)  
# 출력 예: tensor([[ 0.123, -0.456],
#                  [ 1.234,  0.789]])
# → x의 값은 난수로 채워진 2x2 텐서

print(y)  
# 출력 예: tensor([[ 0.987,  1.234],
#                  [-0.654,  0.321]])
# → y의 값은 난수로 채워진 2x2 텐서

print(z)  
# 출력 예: tensor([[0.123*0.987 + (-0.456)*(-0.654), ...],
#                  [..., ...]])
# → z는 x와 y의 행렬 곱셈 결과로, 각 원소는 x의 행과 y의 열의 내적값임

# 3x4 텐서와 4차원 벡터의 행렬-벡터 곱셈 예제
x = torch.randn(3, 4)   # 3x4 텐서, 예: tensor([[...], [...], [...]])
y = torch.randn(4)      # 1차원 텐서, 길이 4 (벡터)

z = x @ y   # 행렬(3x4)와 벡터(4)의 곱 → 결과는 3차원 벡터

print(x)  
# 출력 예: tensor([[x11, x12, x13, x14],
#                  [x21, x22, x23, x24],
#                  [x31, x32, x33, x34]])

print(y)  
# 출력 예: tensor([y1, y2, y3, y4])

print(z)  
# 출력 예: tensor([dot(row1,y), dot(row2,y), dot(row3,y)])
# → 각 원소는 x의 행과 y의 내적 결과

# Broadcasting 예제
# x는 shape (3, 4, 5, 6), y는 shape (4, 1, 6)
x = torch.randn(3, 4, 5, 6)   # 4차원 텐서
y = torch.randn(4, 1, 6)       # 3차원 텐서 → 내부적으로 (1, 4, 1, 6)로 브로드캐스팅됨

z = x + y   # 브로드캐스팅에 의해 y의 값이 x와 동일한 shape (3,4,5,6)으로 확장되어 덧셈 수행

print(x)  
# 출력 예: tensor(shape=(3,4,5,6)) → 실제 값은 난수

print(y)  
# 출력 예: tensor(shape=(4,1,6)) → 브로드캐스팅 전 형태

print(z)  
# 출력 예: tensor(shape=(3,4,5,6)) → x와 y의 덧셈 결과

# 텐서 인덱싱 예제
x = torch.randn(3, 4)  
print(x)  
# 출력 예: tensor([[a, b, c, d],
#                  [e, f, g, h],
#                  [i, j, k, l]])
# → 3행 4열의 난수 텐서

print(x[0])  
# 출력 예: tensor([a, b, c, d])
# → 첫 번째 행(1차원 텐서, 길이 4)

print(x[[0]])  
# 출력 예: tensor([[a, b, c, d]])
# → 첫 번째 행을 2차원 텐서(1x4)로 출력

print(x[0, 1])  
# 출력 예: tensor(b)
# → 첫 번째 행, 두 번째 열의 스칼라 값

print(x[0])  
# 출력 예: tensor([a, b, c, d])
# → 인덱싱 결과는 동일

print(x[0, :])  
# 출력 예: tensor([a, b, c, d])
# → 첫 번째 행 전체 선택

print(x[:, 0])  
# 출력 예: tensor([a, e, i])
# → 모든 행의 첫 번째 열 선택

print(x[0:2, 0:2])  
# 출력 예: tensor([[a, b],
#                  [e, f]])
# → 0행과 1행, 0열과 1열에 해당하는 부분 행렬

print(x[0:2, :])  
# 출력 예: tensor([[a, b, c, d],
#                  [e, f, g, h]])
# → 첫 2행 전체 선택

print(x[:2, :2])  
# 출력 예: tensor([[a, b],
#                  [e, f]])
# → 위와 동일, 슬라이싱 문법의 다른 표현

print(x[:-1, :-1])  
# 출력 예: tensor([[a, b, c],
#                  [e, f, g]])
# → 마지막 행과 마지막 열을 제외한 부분 선택

# 텐서 재구성(reshape) 예제
x = torch.randn(3, 4)  
y = x.reshape(4, 3)  
# x의 12개 원소를 4행 3열로 재배열

z = x.reshape(3, 2, 2)  
# x를 3개의 2x2 행렬로 재구성

w = x.reshape(12)  
# x를 1차원 벡터로 평탄화

z = x.reshape(3, -1, 2)  
# -1은 자동 계산 → 결과는 (3,2,2)와 동일 (12개 원소를 3 x 2 x 2)

print(x)  
# 출력 예: tensor([[x11, x12, x13, x14],
#                  [x21, x22, x23, x24],
#                  [x31, x32, x33, x34]])

print(y)  
# 출력 예: tensor([[x11, x12, x13],
#                  [x14, x21, x22],
#                  [x23, x24, x31],
#                  [x32, x33, x34]])
# → 재구성 순서는 메모리 순서를 따름

print(z)  
# 출력 예: tensor([[[x11, x12],
#                   [x13, x14]],
#                  [[x21, x22],
#                   [x23, x24]],
#                  [[x31, x32],
#                   [x33, x34]]])
# → 3개의 2x2 행렬

print(w)  
# 출력 예: tensor([x11, x12, ..., x34])
# → 12개 원소의 1차원 텐서

print(z[0, 1, 1].item())  
# 출력 예: (x14의 값) → z의 [0,1,1] 위치에 있는 스칼라 값을 파이썬 숫자로 반환

# 텐서의 다양한 함수 예제 (마지막 y 값만 남음)
x = torch.randn(3, 4)  
y = x.cos()    # 코사인 값 계산 → 임시 결과
y = x.sin()    # 사인 값 계산 → 덮어쓰기
y = x.exp()    # 지수함수 계산 → 덮어쓰기
y = x.log()    # 자연로그 계산 → 덮어쓰기 (단, x의 원소가 양수여야 함)
y = x.abs()    # 절댓값 계산 → 덮어쓰기
y = x.sum(dim=1, keepdim=True)  
# 각 행의 합계를 계산하여 (3,1) 텐서로 만듦
y = x.mean(dim=0, keepdim=True)  
# 각 열의 평균을 계산하여 (1,4) 텐서로 만듦 → 최종 y

# 새로운 x 생성 (1차원 난수 벡터)
x = torch.randn(3)  
y = x.norm()  
# x 벡터의 노름(유클리드 길이)를 계산

print(x)  
# 출력 예: tensor([v1, v2, v3]) → 3개의 원소를 가진 벡터

print(y)  
# 출력 예: tensor(s) → x의 노름 (스칼라값)

# 자동 미분(autograd) 예제 1: 단순 선형 연산
x = torch.randn(3, 4, requires_grad=True)  
y = x * 2  
z = y.sum()  
# z = 2 * (x의 모든 원소의 합)

print(x)  
# 출력 예: tensor([[...], [...], [...]]) (requires_grad=True)

print(y)  
# 출력 예: tensor([[...], [...], [...]]) → x의 각 원소에 2가 곱해진 값

print(z)  
# 출력 예: tensor(scalar) → 모든 원소의 합에 2를 곱한 값

z.backward()  
# z에 대해 역전파 실행 → x.grad에 미분값이 저장됨

print(x.grad)  
# 출력 예: tensor([[2., 2., 2., 2.],
#                  [2., 2., 2., 2.],
#                  [2., 2., 2., 2.]])
# → z = 2 * sum(x) 이므로 각 x 원소에 대한 미분은 2

# 자동 미분 예제 2: cos와 exp 연산의 미분
x = torch.randn(3, 4, requires_grad=True)  
y = torch.randn(3, 4, requires_grad=True)  

x1 = x.cos()   # x의 코사인 값 계산
y1 = y.exp()   # y의 지수함수 값 계산

z = x1 + y1    # 두 텐서를 원소별로 더함
z1 = z.sum()   # 스칼라 값으로 만듦

z1.backward()  # 역전파 실행 → x.grad, y.grad에 미분값 저장

print(x.grad)  
# 출력 예: 텐서([[ -sin(x11), -sin(x12), ...], ...])
# → d(cos(x))/dx = -sin(x) 이므로, 각 원소에 대해 -sin(x) 값이 저장됨

print(y.grad)  
# 출력 예: 텐서([[ exp(y11), exp(y12), ...], ...])
# → d(exp(y))/dy = exp(y) 이므로, 각 원소에 대해 exp(y) 값이 저장됨

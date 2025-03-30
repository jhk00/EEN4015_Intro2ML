import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import sys

# 버전 정보 출력
print(f"Python 버전: {sys.version}")
print(f"PyTorch 버전: {torch.__version__}")
print(f"torchvision 버전: {torchvision.__version__}")

# CUDA 상태 확인
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 장치 수: {torch.cuda.device_count()}")
    print(f"CUDA 장치 이름: {torch.cuda.get_device_name(0)}")
    
    # CUDA 버전 확인
    try:
        print(f"CUDA 버전: {torch.version.cuda}")
    except:
        print("CUDA 버전을 확인할 수 없습니다.")

# GPU 강제 설정 시도
try:
    device = torch.device('cuda:0')
    _ = torch.zeros(1).to(device)  # GPU 테스트
    print("GPU 사용이 가능합니다.")
except:
    print("GPU를 사용할 수 없어 CPU로 실행합니다.")
    device = torch.device('cpu')

print(f"사용 중인 장치: {device}")

num_epochs = 3
batch_size = 128
num_classes = 10 # 0~9

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 랜덤 크롭 + 스케일링(80%~100% 크기)
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),  # 약한 회전, 이동, 스케일링
    transforms.ToTensor(),
    # 학습을 위한 텐서 형태로 변환 (C,H,W)
    transforms.Normalize((0.5,), (0.5,)) 
    # 정규화를 통해 값의 범위 조정 (0~255) -> [-1,1]
    # 데이터 분표 표준화, 학습 중 기울기가 적절하게 계산됨
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
	transforms.Normalize((0.5,),(0.5,))
])



train_dataset = datasets.MNIST(root ='./data/', train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_lodaer = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



# 배치 하나 추출
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(f"이미지 shape: {example_data.shape}") 

"""
[128, 1, 28, 28] = [batch_size, channel, Height, Weight]
28×28 픽셀 크기 흑백 이미지
"""

print(f"라벨 shape: {example_targets.shape}")

"""
128개의 이미지 각각에 대한 정답 라벨(0~9 숫자)**이 1차원 텐서로 존재 
"""


class Model(nn.Module):
    def __init__(self,num_classes=10):
        super(Model, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=1, padding=1, bias=False)
        # 더 작은 2x2 커널 사이즈 하려고 했으나 짝수 크기 커널은 중심점이 없어 비추천 (ai response)
        #
        # orgin conv1 -> [3,64...]
        # MNIST -> [1,64..] (흑백)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self,x):
        return self.resnet(x)


model = Model(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)


def train(model, train_loader, criterion, optimzer, device):
    loss_sum = 0.0
    correct = 0.0
    total = 0

    model.train()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return loss_sum / len(train_loader), 100. * correct / total




def evaluate(model, test_loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (iagmes, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return loss_sum / len(test_loader), 100. * correct / total

train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')




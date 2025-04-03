import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
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

num_epochs = 5
batch_size = 128
num_classes = 10  # 0~9

# 데이터 전처리 및 증강
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CNN에 맞게 이미지 크기 조정
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),  # 약한 회전, 이동, 스케일링
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균(0.1307)과 표준편차(0.3081)로 정규화
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CNN에 맞게 이미지 크기 조정
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균과 표준편차로 정규화
])

# 데이터셋 로드
train_dataset = datasets.MNIST(root='./data/', train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 배치 하나 추출 및 확인
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(f"이미지 shape: {example_data.shape}")  # [batch_size, channel, Height, Weight]
print(f"라벨 shape: {example_targets.shape}")

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 입력: 1x32x32, 출력: 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 출력: 32x16x16
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 입력: 32x16x16, 출력: 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 출력: 64x8x8
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 입력: 64x8x8, 출력: 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 출력: 128x4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(), # 128*4*4 = 2048
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

# 모델 초기화 및 장치 설정
model = CNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 학습 함수
def train(model, train_loader, criterion, optimizer, device):
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

# 평가 함수
def evaluate(model, test_loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return loss_sum / len(test_loader), 100. * correct / total

# 학습 및 평가 진행
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

# 학습 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.show()

# 모델 저장
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("모델이 'mnist_cnn.pth'로 저장되었습니다.")
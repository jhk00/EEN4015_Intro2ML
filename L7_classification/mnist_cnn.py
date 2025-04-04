import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import time
from tqdm import tqdm

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

# 하이퍼파라미터 설정
num_epochs = 3  # 에포크 수 (테스트용)
batch_size = 128
num_classes = 10

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CNN에 맞게 이미지 크기 조정
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 데이터셋 로드
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # 첫 번째 Convolutional 블록
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 입력: 1x32x32, 출력: 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 출력: 32x16x16
        )

        # 두 번째 Convolutional 블록
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 입력: 32x16x16, 출력: 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 출력: 64x8x8
        )

        # 세 번째 Convolutional 블록
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 입력: 64x8x8, 출력: 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 출력: 128x4x4
        )

        # 분류기 (Fully Connected 레이어)
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

# 모델 초기화 및 GPU 이동
model = CNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# GPU 사용 확인 - 시작 시 한 번만
print("\n===== GPU 사용 확인 =====")
print(f"모델이 GPU에 있는지: {next(model.parameters()).is_cuda}")

# 첫 번째 배치 데이터 GPU 이동 확인
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    print(f"데이터가 GPU에 있는지: {images.is_cuda}")
    print(f"데이터 형태: {images.shape}")
    # 첫 번째 배치에서 GPU 사용 확인
    with torch.no_grad():
        # 첫 번째 배치로 forward pass 테스트
        outputs = model(images)
        print(f"출력 텐서가 GPU에 있는지: {outputs.is_cuda}")
        print(f"출력 형태: {outputs.shape}")
    break

# 모델 요약 정보 출력
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n모델 파라미터 수: {count_parameters(model):,}")

# 학습 함수 
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 시간 측정
    start_time = time.time()
    
    # 프로그레스 바 추가
    pbar = tqdm(train_loader, desc="Training")
    
    for i, (images, labels) in enumerate(pbar):
        # GPU로 데이터 이동
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 프로그레스 바 업데이트
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    # 시간 측정 종료
    train_time = time.time() - start_time
    print(f"훈련 처리 시간: {train_time:.2f} 초")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, train_time

# 테스트 함수
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# 학습 및 테스트 실행
train_losses = []
train_accs = []
test_losses = []
test_accs = []
times = []

for epoch in range(num_epochs):
    
    # 학습
    train_loss, train_acc, epoch_time = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    times.append(epoch_time)
    
    # 테스트
    test_loss, test_acc = test(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # 에포크 결과 출력
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(f"처리 시간: {epoch_time:.2f} 초")

# 학습 결과 요약
print("\n===== 학습 결과 요약 =====")
print(f"최종 테스트 정확도: {test_accs[-1]:.2f}%")
print(f"평균 에포크 처리 시간: {sum(times)/len(times):.2f} 초")

# 학습 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

print("\n학습이 완료되었습니다!")
print("학습 그래프가 'training_curves.png'로 저장되었습니다.")



import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

def analyze_misclassifications(model, test_loader, device):
    """
    각 클래스별로 어떤 다른 클래스로 잘못 분류되는지 분석합니다.
    """
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_examples = {}  # 클래스별 오분류 예제를 저장할 딕셔너리
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="분석 중"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 전체 예측과 라벨 저장
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 잘못 분류된 경우 저장
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    
                    # 딕셔너리 구조 초기화
                    if true_label not in misclassified_examples:
                        misclassified_examples[true_label] = {}
                    
                    if pred_label not in misclassified_examples[true_label]:
                        misclassified_examples[true_label][pred_label] = []
                    
                    # 최대 5개 예제만 저장
                    if len(misclassified_examples[true_label][pred_label]) < 5:
                        misclassified_examples[true_label][pred_label].append(images[i].cpu())
    
    # 1. 혼동 행렬 생성 및 시각화
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('예측된 클래스')
    plt.ylabel('실제 클래스')
    plt.title('혼동 행렬')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 2. 각 클래스별 오분류 패턴 출력
    print("\n각 클래스별 주요 오분류 패턴:")
    for true_label in sorted(misclassified_examples.keys()):
        print(f"\n실제 클래스 {true_label}:")
        # 오분류 횟수 계산
        misclass_counts = {}
        for pred_label in misclassified_examples[true_label]:
            row_sum = cm[true_label].sum()
            if row_sum > 0:  # 0으로 나누기 방지
                misclass_counts[pred_label] = cm[true_label, pred_label]
        
        # 오분류 횟수 기준으로 정렬
        sorted_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
        
        for pred_label, count in sorted_misclass:
            percentage = (count / cm[true_label].sum()) * 100
            print(f"  → 클래스 {pred_label}로 {count}번 오분류됨 ({percentage:.1f}%)")
    
    # 3. 오분류 예제 시각화
    for true_label in sorted(misclassified_examples.keys()):
        most_common_errors = sorted(
            [(pred, len(examples)) for pred, examples in misclassified_examples[true_label].items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        if most_common_errors:  # 오분류가 있는 경우에만
            # 가장 많이 오분류된 클래스 3개만
            top_errors = most_common_errors[:min(3, len(most_common_errors))]
            
            fig, axes = plt.subplots(len(top_errors), 5, figsize=(15, 3*len(top_errors)))
            if len(top_errors) == 1:
                axes = [axes]  # 1행인 경우 2D 배열로 변환
            
            for i, (pred_label, _) in enumerate(top_errors):
                examples = misclassified_examples[true_label][pred_label]
                
                for j, img in enumerate(examples[:5]):  # 최대 5개 예제 표시
                    if j < 5:  # 배열 범위 확인
                        ax = axes[i][j]
                        ax.imshow(img.squeeze().numpy(), cmap='gray')
                        ax.set_title(f'진짜: {true_label}, 예측: {pred_label}')
                        ax.axis('off')
                
                # 예제가 5개 미만인 경우 남은 축 숨기기
                for j in range(len(examples), 5):
                    axes[i][j].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'class_{true_label}_misclassified.png')
            plt.show()
    
    print("\n분석 결과가 이미지 파일로 저장되었습니다.")
    return cm, misclassified_examples

# 모델 학습 완료 후 이 함수를 호출
# analyze_misclassifications(model, test_loader, device)

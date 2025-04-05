import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import wandb  # WandB 라이브러리 추가

# WandB API 키 설정 및 초기화
wandb.login(key="ef091b9abcea3186341ddf8995d62bde62d7469e")
wandb.init(project="mnist-cnn", name="focal-loss-with_arguments")  # 프로젝트와 실험 이름 설정

# 하이퍼파라미터 기록
config = {
    "num_epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": 10,
    "focal_loss_gamma": 2.0,  # Focal Loss의 감마 매개변수
    "focal_loss_alpha": 1.0   # Focal Loss의 알파 매개변수
}
wandb.config.update(config)  # WandB에 하이퍼파라미터 기록


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


num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
num_classes = config["num_classes"]  # 0~9

# 데이터 전처리 및 증강 (더 강화된 증강 적용)
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CNN에 맞게 이미지 크기 조정
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.9, 1.1), shear=10),  # 강화된 회전, 이동, 스케일링
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 원근 변환 추가
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1),  # ToTensor 이후에 적용 (Tensor에서만 작동)
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


# Focal Loss 구현
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 모델 초기화 및 장치 설정
model = CNN(num_classes=num_classes).to(device)
wandb.watch(model, log="all")  # WandB가 모델 가중치와 그래디언트를 추적하도록 설정

# Focal Loss 사용으로 변경
criterion = FocalLoss(alpha=config["focal_loss_alpha"], gamma=config["focal_loss_gamma"])
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

# 학습률 스케줄러 추가
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

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

    # 예측 결과와 실제 레이블을 저장할 리스트
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()

            # 예측 및 신뢰도
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 예측 및 레이블 저장
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # WandB에 혼동 행렬 로깅
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=[str(i) for i in range(num_classes)]
    )})

    return loss_sum / len(test_loader), 100. * correct / total

# 학습 및 평가 진행
train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_acc = 0.0

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # 학습률 스케줄러 업데이트
    scheduler.step(test_loss)

    # 최고 성능 모델 저장
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'mnist_focal_best.pth')
        print(f"새로운 최고 정확도: {best_acc:.2f}% - 모델 저장됨")

    print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # WandB에 로그 기록
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

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

# WandB에 그래프 이미지 저장
wandb.log({"learning_curves": wandb.Image(plt)})

plt.show()

# 최종 모델 저장
torch.save(model.state_dict(), 'mnist_focal_final.pth')
print("최종 모델이 'mnist_focal_final.pth'로 저장되었습니다.")
print(f"최고 정확도 모델은 'mnist_focal_best.pth'로 저장되었습니다. (정확도: {best_acc:.2f}%)")

# 최종 테스트에서 오분류된 이미지들 수집
print("최종 테스트에서 오분류된 이미지 수집 중...")
model.load_state_dict(torch.load('mnist_focal_best.pth'))  # 최고 성능 모델 로드
model.eval()

final_misclassified_images = []
final_misclassified_info = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

        # 오분류된 이미지 찾기
        misclassified = ~predicted.eq(labels)
        if misclassified.sum() > 0:
            misclassified_indices = misclassified.nonzero().squeeze()
            if misclassified_indices.dim() == 0:  # 단일 인덱스인 경우
                misclassified_indices = misclassified_indices.unsqueeze(0)

            for idx in misclassified_indices:
                if len(final_misclassified_images) < 100:  # 최대 100개까지만 저장
                    final_misclassified_images.append(images[idx].cpu())
                    final_misclassified_info.append({
                        'true': labels[idx].item(),
                        'pred': predicted[idx].item(),
                        'conf': confidence[idx].item()
                    })

# 모든 오분류 이미지를 하나의 파일로 저장 (레이블 정보 없이)
if len(final_misclassified_images) > 0:
    n_images = len(final_misclassified_images)
    rows = int(n_images ** 0.5) + 1
    cols = (n_images // rows) + 1

    plt.figure(figsize=(cols * 2.5, rows * 2.5))

    for i, (image, _) in enumerate(zip(final_misclassified_images, final_misclassified_info)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.axis('off')

    # 저장 경로 설정
    final_save_path = 'final_misclassified.png'
    plt.tight_layout()
    plt.savefig(final_save_path, dpi=200)
    print(f"최종 오분류 이미지가 '{final_save_path}'에 저장되었습니다.")
    plt.close()

    # WandB에 통합 이미지 파일도 로깅
    wandb.log({"final_misclassified": wandb.Image(final_save_path)})

# WandB에 모델 파일 업로드
wandb.save('mnist_focal_final.pth')
wandb.save('mnist_focal_best.pth')
wandb.save('final_misclassified.png')

# WandB 세션 종료
wandb.finish()
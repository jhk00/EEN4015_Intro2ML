import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import numpy as np
import wandb  # WandB 라이브러리 추가

# WandB API 키 설정 및 초기화
wandb.login(key="ef091b9abcea3186341ddf8995d62bde62d7469e")
wandb.init(project="mnist-cnn", name="combined-augmentation-paper-fixed")  # 프로젝트와 실험 이름 설정

# 하이퍼파라미터 기록
config = {
    "num_epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": 10,
    "focal_loss_gamma": 2.0,  # Focal Loss의 감마 매개변수
    "focal_loss_alpha": 1.0,   # Focal Loss의 알파 매개변수
    "augmentation_probability": 0.7  # 데이터 증강 적용 확률
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

# PIL Image와 호환되는 커스텀 증강 변환 클래스 정의
class BackslashTransform:
    def __init__(self, factor=0.3):
        self.factor = factor
        
    def __call__(self, img):
        # PIL Image를 받아서 처리
        return transforms.functional.affine(
            img, 
            angle=0, 
            translate=(0, 0), 
            scale=1.0, 
            shear=(self.factor * 45, 0)  # 백슬래시(\\) 방향 기울기
        )

class SlashTransform:
    def __init__(self, factor=0.3):
        self.factor = factor
        
    def __call__(self, img):
        # PIL Image를 받아서 처리
        return transforms.functional.affine(
            img, 
            angle=0, 
            translate=(0, 0), 
            scale=1.0, 
            shear=(0, self.factor * 45)  # 슬래시(/) 방향 기울기
        )

class MultiplicationTransform:
    def __init__(self, factor=0.2):
        self.factor = factor
        self.backslash = BackslashTransform(factor=self.factor)
        self.slash = SlashTransform(factor=self.factor)
        
    def __call__(self, img):
        # 먼저 backslash 적용 후 slash 적용
        img = self.backslash(img)
        return self.slash(img)

class GreaterThanTransform:
    def __init__(self, factor=0.25):
        self.factor = factor
        
    def __call__(self, img):
        return transforms.functional.affine(
            img, 
            angle=0, 
            translate=(0, 0), 
            scale=1.0, 
            shear=(0, self.factor * 45)  # '>' 모양을 위한 오른쪽 방향 기울기
        )

class OTransform:
    def __init__(self, distortion=0.3):
        self.distortion = distortion
        
    def __call__(self, img):
        # PIL Image의 크기 가져오기
        width, height = img.size
        
        # 'O' 모양 변형은 perspective 변환으로 근사
        startpoints = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]  # 원본 이미지 모서리
        endpoints = [
            (int(0 + width * self.distortion), int(0 + height * self.distortion)),  # 좌상단
            (int(width - 1 - width * self.distortion), int(0 + height * self.distortion)),  # 우상단 
            (int(0 + width * self.distortion), int(height - 1 - height * self.distortion)),  # 좌하단
            (int(width - 1 - width * self.distortion), int(height - 1 - height * self.distortion))  # 우하단
        ]
        
        return transforms.functional.perspective(img, startpoints, endpoints, fill=0)

# 다중 증강 기법을 순차적으로 적용하는 복합 변환 클래스
class CombinedAugmentation:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list
    
    def __call__(self, img):
        for transform in self.transforms_list:
            img = transform(img)
        return img

# PIL Image 기반 변환 객체 생성 
backslash_transform = BackslashTransform(factor=0.3)
slash_transform = SlashTransform(factor=0.3)
greater_than_transform = GreaterThanTransform(factor=0.25)
o_transform = OTransform(distortion=0.3)
rotation_transform = transforms.RandomRotation(degrees=15)
affine_transform = transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.9, 1.1), shear=10)
perspective_transform = transforms.RandomPerspective(distortion_scale=0.2, p=1.0)

# 논문 실험처럼 다양한 2중 증강 조합
combined_augmentations = [
    # 논문에서 언급된 조합
    CombinedAugmentation([backslash_transform, slash_transform]),  # 조합 1: \와 / 조합
    CombinedAugmentation([backslash_transform, greater_than_transform]),  # 조합 2: \와 > 조합
    CombinedAugmentation([slash_transform, o_transform]),  # 조합 3: /와 O 조합
    
    # 추가 실험용 조합
    CombinedAugmentation([rotation_transform, backslash_transform]),  # 조합 4: 회전과 \ 조합
    CombinedAugmentation([affine_transform, o_transform]),  # 조합 5: 아핀 변환과 O 조합
    CombinedAugmentation([perspective_transform, slash_transform]),  # 조합 6: 원근 변환과 / 조합
]

# 데이터 전처리 및 증강 - 다양한 증강 기법 적용
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CNN에 맞게 이미지 크기 조정
    transforms.RandomChoice([
        # 원본 이미지 (증강 없음)
        transforms.Lambda(lambda x: x),
        
        # 단일 증강 기법 (약 30% 확률)
        backslash_transform, 
        slash_transform, 
        MultiplicationTransform(factor=0.2),
        greater_than_transform,
        o_transform,
        affine_transform,
        
        # 복합 증강 기법 (약 70% 확률로 2종류 이상 증강 적용)
        *combined_augmentations  # 다양한 복합 증강 기법
    ]),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균과 표준편차로 정규화
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
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# CNN 모델 정의 - 3개의 convolutional layer와 더 깊은 분류기
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
def evaluate(model, data_loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    # 예측 결과와 실제 레이블을 저장할 리스트
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
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

    return loss_sum / len(data_loader), 100. * correct / total, all_preds, all_labels

# 학습 및 평가 진행
train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_acc = 0.0
best_model_path = 'mnist_combined_aug_best.pth'

print("===== 학습 시작 =====")
for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 현재 학습률
    current_lr = optimizer.param_groups[0]['lr']

    # 매 에폭마다 테스트 셋으로 평가
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # 혼동 행렬 로깅
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=test_labels,
            preds=test_preds,
            class_names=[str(i) for i in range(num_classes)]
        ),
        "test_accuracy": test_acc,
        "test_loss": test_loss
    })
    
    # 모델 저장 (테스트 정확도 기준)
    save_message = ""
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        save_message = f" - 새로운 최고 정확도! 모델 저장됨"
    
    print(f'Epoch: {epoch+1}/{num_epochs}    Train Loss: {train_loss:.4f}    Train Acc: {train_acc:.2f}%    Test Loss: {test_loss:.4f}    Test Acc: {test_acc:.2f}%{save_message}')

    # WandB에 로그 기록
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "learning_rate": current_lr
    })

# 최종 테스트 평가
print("\n===== 최종 평가 =====")
# 최고 성능 모델 로드
model.load_state_dict(torch.load(best_model_path))
final_test_loss, final_test_acc, final_test_preds, final_test_labels = evaluate(model, test_loader, criterion, device)
print(f"최종 테스트 정확도: {final_test_acc:.2f}%")

# WandB에 최종 결과 로깅
wandb.log({
    "final_test_accuracy": final_test_acc,
    "final_test_loss": final_test_loss,
    "final_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=final_test_labels,
        preds=final_test_preds,
        class_names=[str(i) for i in range(num_classes)]
    )
})

# 학습 결과 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(range(num_epochs), test_accs, label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()

# WandB에 그래프 이미지 저장
wandb.log({"learning_curves": wandb.Image(plt)})

plt.savefig('learning_curves_combined_aug.png')
plt.show()

# 최종 모델 저장
torch.save(model.state_dict(), 'mnist_combined_aug_final.pth')
print("최종 모델이 'mnist_combined_aug_final.pth'로 저장되었습니다.")
print(f"최고 정확도 모델은 '{best_model_path}'로 저장되었습니다. (테스트 정확도: {best_acc:.2f}%)")

# 증강 시각화 함수 - PIL Image 관련 오류 수정
def visualize_augmentations():
    # 원본 MNIST에서 숫자 샘플 가져오기
    # 이미지 변환을 적용하지 않은 데이터셋 생성
    original_dataset = datasets.MNIST(root='./data/', train=True, download=False, transform=None)
    
    # 각 숫자 유형별 샘플 이미지 선택
    samples = []
    for digit in range(5):  # 0-4 숫자만 표시
        indices = (original_dataset.targets == digit).nonzero().squeeze()
        samples.append(original_dataset[indices[0]][0])  # PIL Image
    
    # 증강 변환 정의
    augmentations = [
        ("원본", lambda x: x),
        ("백슬래시(\\)", BackslashTransform(factor=0.3)),
        ("슬래시(/)", SlashTransform(factor=0.3)),
        ("곱셈기호(×)", MultiplicationTransform(factor=0.2)),
        ("부등호(>)", GreaterThanTransform(factor=0.25)),
        ("O자형", OTransform(distortion=0.3)),
        ("\\+/", combined_augmentations[0]),
        ("\\+>", combined_augmentations[1]),
        ("/+O", combined_augmentations[2])
    ]
    
    # 시각화
    fig, axes = plt.subplots(len(samples), len(augmentations), figsize=(18, 10))
    
    for i, sample in enumerate(samples):
        for j, (aug_name, aug_transform) in enumerate(augmentations):
            # 원본 이미지 복사 (PIL Image)
            img = sample.copy()
            
            # 증강 적용
            if j > 0:  # 첫 번째 열은 원본
                img = aug_transform(img)
            
            # 이미지 표시
            if i == 0:  # 첫 번째 행에만 제목 표시
                axes[i, j].set_title(aug_name, fontsize=10)
            
            # PIL Image를 numpy 배열로 변환하여 표시
            img_array = np.array(img)
            axes[i, j].imshow(img_array, cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150)
    
    # WandB에 이미지 로깅
    wandb.log({"augmentation_examples": wandb.Image('augmentation_examples.png')})
    
    plt.show()
    return fig

# 증강 시각화 실행
try:
    fig = visualize_augmentations()
    print("증강 시각화 완료")
except Exception as e:
    print(f"증강 시각화 중 오류 발생: {e}")

# 클래스별 정확도 분석
def analyze_class_accuracy():
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 클래스별 정확도 계산
    class_accuracy = [100 * correct / total for correct, total in zip(class_correct, class_total)]
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), class_accuracy, color='skyblue')
    plt.xlabel('숫자 클래스')
    plt.ylabel('정확도 (%)')
    plt.title('클래스별 테스트 정확도')
    plt.xticks(range(num_classes))
    plt.ylim(90, 100)  # MNIST는 일반적으로 90% 이상의 정확도를 보임
    plt.grid(axis='y', alpha=0.3)
    
    for i, acc in enumerate(class_accuracy):
        plt.text(i, acc - 0.5, f'{acc:.1f}%', ha='center', va='top')
    
    plt.savefig('class_accuracy.png')
    
    # WandB에 로깅
    wandb.log({
        "class_accuracy": wandb.Image('class_accuracy.png'),
        "class_accuracy_values": {f"class_{i}_acc": acc for i, acc in enumerate(class_accuracy)}
    })
    
    plt.show()
    
    return class_accuracy

# 클래스별 정확도 분석 실행
class_accuracy = analyze_class_accuracy()
print("\n클래스별 정확도:")
for i, acc in enumerate(class_accuracy):
    print(f"숫자 {i}: {acc:.2f}%")

# 오분류된 이미지 분석
print("\n===== 오분류 이미지 분석 =====")
model.load_state_dict(torch.load(best_model_path))
model.eval()

misclassified_images = []
misclassified_info = []

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
                if len(misclassified_images) < 25:  # 최대 25개까지만 저장
                    misclassified_images.append(images[idx].cpu())
                    misclassified_info.append({
                        'true': labels[idx].item(),
                        'pred': predicted[idx].item(),
                        'conf': confidence[idx].item()
                    })

# 오분류 이미지 시각화
if len(misclassified_images) > 0:
    n_images = len(misclassified_images)
    rows = int(np.ceil(np.sqrt(n_images)))
    cols = int(np.ceil(n_images / rows))

    plt.figure(figsize=(cols * 2.5, rows * 2.5))

    for i in range(n_images):
        plt.subplot(rows, cols, i + 1)
        img = misclassified_images[i].squeeze().numpy()
        # 정규화 복원
        img = img * 0.3081 + 0.1307
        plt.imshow(img, cmap='gray')
        info = misclassified_info[i]
        plt.title(f"T: {info['true']}, P: {info['pred']}\nConf: {info['conf']:.2f}")
        plt.axis('off')

    # 저장 경로 설정
    misclassified_path = 'misclassified_examples.png'
    plt.tight_layout()
    plt.savefig(misclassified_path, dpi=150)
    
    # WandB에 이미지 로깅
    wandb.log({"misclassified_examples": wandb.Image(misclassified_path)})
    
    plt.show()

# 실험 결과 요약
print("\n===== 실험 결과 요약 =====")
print(f"최종 테스트 정확도: {final_test_acc:.2f}%")
print(f"학습 에폭 수: {num_epochs}")
print(f"배치 크기: {batch_size}")
print(f"학습률: {config['learning_rate']}")
print(f"Focal Loss gamma: {config['focal_loss_gamma']}")
print(f"Focal Loss alpha: {config['focal_loss_alpha']}")
print("\n다양한 커스텀 증강 기법 적용 (스케줄러 없이 고정 학습률 사용)")

# WandB 세션 종료
wandb.finish()
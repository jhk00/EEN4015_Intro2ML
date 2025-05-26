import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import time
from torchmetrics.classification import MulticlassAveragePrecision

class EfficientNetTeacher(nn.Module):
    """EfficientNet-B2 Noisy Student 기반 멀티태스크 Teacher"""
    def __init__(self, num_domains=4, num_classes=65):
        super(EfficientNetTeacher, self).__init__()
        
        #  EfficientNet-B2 Noisy Student 
        self.backbone = timm.create_model(
            'tf_efficientnet_b2_ns',  # Noisy Student 버전
            pretrained=True,
            num_classes=0  # 분류 헤드 제거
        )
        
        feature_dim = self.backbone.num_features  # 1408 for B2
        
        # 멀티태스크 헤드 (Student보다 약간 큰 용량)
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_domains)
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # EfficientNet 특징 추출
        features = self.backbone(x)
        
        # 멀티태스크 출력
        domain_out = self.domain_head(features)
        class_out = self.class_head(features)
        
        return domain_out, class_out

class MultiTaskKDLoss(nn.Module):
    """멀티태스크 Knowledge Distillation Loss"""
    def __init__(self, domain_alpha=0.8, class_alpha=0.6, domain_temp=5.0, class_temp=3.0):
        super(MultiTaskKDLoss, self).__init__()
        
        # 도메인 분류 설정 (더 부드럽게 - 오버피팅 방지)
        self.domain_alpha = domain_alpha  # KD 비중 높게
        self.domain_temp = domain_temp    # 부드러운 분포
        
        # 클래스 분류 설정
        self.class_alpha = class_alpha
        self.class_temp = class_temp
        
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_domain_logits, student_class_logits, 
                teacher_domain_logits, teacher_class_logits,
                domain_labels, class_labels):
        """KD Loss 계산"""
        
        #  도메인 분류 KD 
        student_domain_soft = F.log_softmax(student_domain_logits / self.domain_temp, dim=1)
        teacher_domain_soft = F.softmax(teacher_domain_logits / self.domain_temp, dim=1)
        
        domain_kd_loss = self.kl_div(student_domain_soft, teacher_domain_soft) * (self.domain_temp ** 2)
        domain_hard_loss = self.ce_loss(student_domain_logits, domain_labels)
        domain_total_loss = self.domain_alpha * domain_kd_loss + (1 - self.domain_alpha) * domain_hard_loss
        
        # 클래스 분류 KD
        student_class_soft = F.log_softmax(student_class_logits / self.class_temp, dim=1)
        teacher_class_soft = F.softmax(teacher_class_logits / self.class_temp, dim=1)
        
        class_kd_loss = self.kl_div(student_class_soft, teacher_class_soft) * (self.class_temp ** 2)
        class_hard_loss = self.ce_loss(student_class_logits, class_labels)
        class_total_loss = self.class_alpha * class_kd_loss + (1 - self.class_alpha) * class_hard_loss
        
        return {
            'domain_total_loss': domain_total_loss,
            'domain_kd_loss': domain_kd_loss,
            'domain_hard_loss': domain_hard_loss,
            'class_total_loss': class_total_loss,
            'class_kd_loss': class_kd_loss,
            'class_hard_loss': class_hard_loss
        }

def pretrain_teacher(teacher, trainloader, testloader, device, epochs=20):
    """Teacher 사전 훈련"""
    print("🎓 EfficientNet Teacher 사전 훈련 시작...")
    
    teacher.train()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    domain_criterion = nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss()
    
    best_performance = 0.0
    
    for epoch in range(epochs):
        # 훈련
        teacher.train()
        running_loss = 0.0
        domain_correct = 0
        class_correct = 0
        total = 0
        
        for i, (inputs, domain_labels, class_labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            domain_labels = domain_labels.to(device)
            class_labels = class_labels.to(device)
            
            optimizer.zero_grad()
            
            domain_outputs, class_outputs = teacher(inputs)
            
            # 균등한 가중치로 Teacher 훈련
            domain_loss = domain_criterion(domain_outputs, domain_labels)
            class_loss = class_criterion(class_outputs, class_labels)
            loss = 0.5 * domain_loss + 0.5 * class_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, domain_preds = domain_outputs.max(1)
            domain_correct += domain_preds.eq(domain_labels).sum().item()
            
            _, class_preds = class_outputs.max(1)
            class_correct += class_preds.eq(class_labels).sum().item()
            
            total += inputs.size(0)
            
            if i % 20 == 0:
                print(f'Teacher Epoch [{epoch+1}/{epochs}], Batch [{i+1}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(trainloader)
        domain_acc = 100.0 * domain_correct / total
        class_acc = 100.0 * class_correct / total
        avg_acc = (domain_acc + class_acc) / 2
        
        print(f'Teacher Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Domain: {domain_acc:.2f}%, Class: {class_acc:.2f}%')
        
        scheduler.step(avg_acc)
        
        if avg_acc > best_performance:
            best_performance = avg_acc
            torch.save(teacher.state_dict(), 'efficientnet_teacher_best.pth')
            print(f'🏆 Teacher 최고 성능: {best_performance:.2f}%')
    
    # 최고 성능 모델 로드
    teacher.load_state_dict(torch.load('efficientnet_teacher_best.pth'))
    print(f"✅ Teacher 훈련 완료! 최고 성능: {best_performance:.2f}%")
    return teacher



def train_with_kd(model, teacher, trainloader, kd_loss_fn, optimizer, device, epoch, config):
    """KD 적용 훈련 함수 (기존 train 함수 대체)"""
    model.train()
    teacher.eval()  # Teacher는 항상 평가 모드
    
    start_time = time.time()
    running_losses = {
        'total': 0.0, 'domain_total': 0.0, 'class_total': 0.0,
        'domain_kd': 0.0, 'domain_hard': 0.0, 'class_kd': 0.0, 'class_hard': 0.0
    }
    
    domain_correct = 0
    class_correct = 0
    total = 0
    
    # mAP 계산기
    domain_map = MulticlassAveragePrecision(num_classes=config["num_domains"], average='macro').to(device)
    class_map = MulticlassAveragePrecision(num_classes=config["num_classes"], average='macro').to(device)
    
    for i, (inputs, domain_labels, class_labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        domain_labels = domain_labels.to(device)
        class_labels = class_labels.to(device)
        
        optimizer.zero_grad()
        
        # 🎓 Teacher 예측 (no grad)
        with torch.no_grad():
            teacher_domain_outputs, teacher_class_outputs = teacher(inputs)
        
        # 🎒 Student 예측
        student_domain_outputs, student_class_outputs = model(inputs)
        
        # 🔥 KD Loss 계산
        loss_dict = kd_loss_fn(
            student_domain_outputs, student_class_outputs,
            teacher_domain_outputs, teacher_class_outputs,
            domain_labels, class_labels
        )
        
        # 전체 손실 (기존 가중치 적용)
        total_loss = (config["domain_weight"] * loss_dict['domain_total_loss'] + 
                     config["class_weight"] * loss_dict['class_total_loss'])
        
        total_loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        for key in running_losses:
            if key == 'total':
                running_losses[key] += total_loss.item()
            else:
                running_losses[key] += loss_dict[f'{key}_loss'].item()
        
        # 정확도 계산
        _, domain_preds = student_domain_outputs.max(1)
        domain_correct += domain_preds.eq(domain_labels).sum().item()
        
        _, class_preds = student_class_outputs.max(1)
        class_correct += class_preds.eq(class_labels).sum().item()
        
        total += inputs.size(0)
        
        # mAP 업데이트
        domain_map.update(student_domain_outputs, domain_labels)
        class_map.update(student_class_outputs, class_labels)
        
        if (i + 1) % 20 == 0:
            print(f'KD Epoch [{epoch+1}], Batch [{i+1}/{len(trainloader)}]')
            print(f'Total: {total_loss.item():.4f} | Domain KD: {loss_dict["domain_kd_loss"].item():.4f} | Class KD: {loss_dict["class_kd_loss"].item():.4f}')
    
    # 에폭 통계
    epoch_losses = {key: val / len(trainloader) for key, val in running_losses.items()}
    domain_accuracy = 100.0 * domain_correct / total
    class_accuracy = 100.0 * class_correct / total
    domain_map_value = domain_map.compute().item()
    class_map_value = class_map.compute().item()
    
    train_time = time.time() - start_time
    
    print(f'KD Train Epoch {epoch+1}:')
    print(f'Total Loss: {epoch_losses["total"]:.4f} | Domain Acc: {domain_accuracy:.2f}% | Class Acc: {class_accuracy:.2f}%')
    print(f'Domain mAP: {domain_map_value:.4f} | Class mAP: {class_map_value:.4f}')
    print(f'KD Losses - Domain: {epoch_losses["domain_kd"]:.4f} | Class: {epoch_losses["class_kd"]:.4f}')
    print(f'Time: {train_time:.2f}s\n')
    
    return (epoch_losses["total"], epoch_losses["domain_total"], epoch_losses["class_total"], 
            domain_accuracy, class_accuracy, domain_map_value, class_map_value, epoch_losses)
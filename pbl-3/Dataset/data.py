
import torch
import torch.nn as nn
import sys
from torch.utils.data import Dataset
from PIL import Image
import os

class OfficeHomeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.class_labels = []
        self.domain_labels = []
        
        # 도메인 매핑
        self.domains = {'Art': 0, 'Clipart': 1, 'Product': 2, 'Real_World': 3}
        
        # 클래스 매핑 (디렉토리에서 자동 감지)
        self.classes = {}
        class_idx = 0
        
        # 첫 번째 도메인에서 클래스 목록 초기화
        first_domain = next(iter(self.domains))
        first_domain_dir = os.path.join(root_dir, first_domain)
        if os.path.isdir(first_domain_dir):
            for class_name in os.listdir(first_domain_dir):
                if os.path.isdir(os.path.join(first_domain_dir, class_name)):
                    if class_name not in self.classes:
                        self.classes[class_name] = class_idx
                        class_idx += 1
        
        # 모든 도메인과 클래스에서 이미지 로드
        for domain, domain_idx in self.domains.items():
            domain_dir = os.path.join(root_dir, domain)
            if not os.path.isdir(domain_dir):
                continue
            
            for class_name in os.listdir(domain_dir):
                class_path = os.path.join(domain_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                # 클래스 ID 가져오기
                class_idx = self.classes.get(class_name)
                if class_idx is None:
                    print(f"경고: 알 수 없는 클래스 '{class_name}' in {domain}")
                    continue
                
                # 이미지 파일 추가
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.domain_labels.append(domain_idx)
                        self.class_labels.append(class_idx)
        
        print(f"총 {len(self.images)} 이미지, {len(self.classes)} 클래스, {len(self.domains)} 도메인을 로드했습니다.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        domain_label = self.domain_labels[idx]
        class_label = self.class_labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, domain_label, class_label
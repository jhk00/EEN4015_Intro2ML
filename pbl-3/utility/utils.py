import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

class DualMAPEarlyStopping:
    """Domain mAP와 Class mAP 두 지표 모두 patience 이후로 개선되지 않으면 학습을 조기 중지합니다."""
    def __init__(self, patience=30, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, max_epochs=float('inf')):
        """
        Args:
            patience (int): 마지막 mAP 개선 이후 기다릴 에포크 수
                           기본값: 30
            verbose (bool): True일 경우, 각 mAP 개선마다 메시지 출력
                           기본값: False
            delta (float): 개선으로 인정할 최소 변화량
                           기본값: 0
            path (str): 체크포인트가 저장될 경로
                           기본값: 'checkpoint.pt'
            trace_func (function): 출력 함수
                           기본값: print
            max_epochs (int): 강제 종료할 최대 에폭 수
                           기본값: 무한대
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.domain_best_score = None
        self.class_best_score = None
        self.early_stop = False
        self.domain_map_max = -np.Inf  # 최고 도메인 mAP를 음의 무한대로 초기화
        self.class_map_max = -np.Inf   # 최고 클래스 mAP를 음의 무한대로 초기화
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.max_epochs = max_epochs   # 최대 에폭 수 제한
        self.domain_best_epoch = 0     # 도메인 mAP 최고 성능을 기록한 에폭
        self.class_best_epoch = 0      # 클래스 mAP 최고 성능을 기록한 에폭
        self.best_epoch = 0            # 전체 최고 성능을 기록한 에폭 (두 mAP 중 하나라도 개선된 경우)
    
    def __call__(self, domain_map, class_map, model, epoch):
        # 에폭 제한 확인
        if epoch >= self.max_epochs:
            self.early_stop = True
            if self.verbose:
                self.trace_func(f"최대 에폭 ({self.max_epochs})에 도달했습니다. 훈련을 중단합니다.")
            return
            
        domain_improved = False
        class_improved = False
        
        # 도메인 mAP 검사
        if self.domain_best_score is None:
            self.domain_best_score = domain_map
            self.domain_best_epoch = epoch
            domain_improved = True
        elif domain_map > self.domain_best_score + self.delta:
            self.domain_best_score = domain_map
            self.domain_best_epoch = epoch
            domain_improved = True
            if self.verbose:
                self.trace_func(f'Domain mAP improved ({self.domain_map_max:.4f} --> {domain_map:.4f}).')
        
        # 클래스 mAP 검사
        if self.class_best_score is None:
            self.class_best_score = class_map
            self.class_best_epoch = epoch
            class_improved = True
        elif class_map > self.class_best_score + self.delta:
            self.class_best_score = class_map
            self.class_best_epoch = epoch
            class_improved = True
            if self.verbose:
                self.trace_func(f'Class mAP improved ({self.class_map_max:.4f} --> {class_map:.4f}).')
        
        # 둘 중 하나라도 개선되었으면 모델 저장 및 카운터 리셋
        if domain_improved or class_improved:
            self.save_best_model(domain_map, class_map, model)
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping 카운터: {self.counter} / {self.patience} '
                               f'(Domain 최고: {self.domain_best_score:.4f}, Class 최고: {self.class_best_score:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func(f'EarlyStopping: {self.patience}번의 에폭 동안 Domain mAP와 Class mAP가 모두 개선되지 않았습니다.')
    
    def save_best_model(self, domain_map, class_map, model):
        '''mAP가 증가하면 모델을 저장합니다.'''
        if self.verbose:
            self.trace_func(f'Performance improved. Saving model ...')
            self.trace_func(f'Domain mAP: {self.domain_map_max:.4f} --> {domain_map:.4f}, Class mAP: {self.class_map_max:.4f} --> {class_map:.4f}')
        torch.save(model.state_dict(), self.path)
        self.domain_map_max = max(domain_map, self.domain_map_max)
        self.class_map_max = max(class_map, self.class_map_max)
        










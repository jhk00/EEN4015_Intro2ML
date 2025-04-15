import numpy as np
import torch

class EarlyStopping:
    """지정된 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지합니다."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): 마지막 validation loss 개선 이후 기다릴 에포크 수
                            기본값: 7
            verbose (bool): True일 경우, 각 validation loss 개선마다 메시지 출력
                            기본값: False
            delta (float): 개선으로 인정할 최소 변화량
                            기본값: 0
            path (str): 체크포인트가 저장될 경로
                            기본값: 'checkpoint.pt'
            trace_func (function): 출력 함수
                            기본값: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping 카운터: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장합니다.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)  # self.path 사용
        self.val_loss_min = val_loss


class AccuracyEarlyStopping:
    """지정된 patience 이후로 accuracy가 개선되지 않으면 학습을 조기 중지합니다."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, max_epochs=float('inf')):
        """
        Args:
            patience (int): 마지막 accuracy 개선 이후 기다릴 에포크 수
                            기본값: 7
            verbose (bool): True일 경우, 각 accuracy 개선마다 메시지 출력
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
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf  # 최고 정확도를 음의 무한대로 초기화
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.max_epochs = max_epochs  # 최대 에폭 수 제한
        self.best_epoch = 0  # 최고 성능을 기록한 에폭
    
    def __call__(self, val_acc, model, epoch):
        # 에폭 제한 확인
        if epoch >= self.max_epochs:
            self.early_stop = True
            if self.verbose:
                self.trace_func(f"최대 에폭 ({self.max_epochs})에 도달했습니다. 훈련을 중단합니다.")
            return
            
        score = val_acc  # 정확도는 높을수록 좋으므로 그대로 사용
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping 카운터: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_acc, model)
            self.counter = 0
    
    def save_checkpoint(self, val_acc, model):
        '''정확도가 증가하면 모델을 저장합니다.'''
        if self.verbose:
            self.trace_func(f'Accuracy improved ({self.val_acc_max:.2f}% --> {val_acc:.2f}%). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc
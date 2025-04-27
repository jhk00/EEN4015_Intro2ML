import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

class AccuracyEarlyStopping:
    """지정된 patience 이후로 accuracy가 개선되지 않으면 학습을 조기 중지합니다."""
    def __init__(self, patience=30, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, max_epochs=float('inf')):
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

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]




class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

import torch.nn as nn
from torch.nn import functional as F

import torch


class PSKD(nn.Module):
	def __init__(self):
		super(PSKD, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, output, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(output)
		loss = (- targets * log_probs).mean(0).sum()
		return loss        
        


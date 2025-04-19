import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ShakeDrop(nn.Module):
    """ShakeDrop implementation using standard PyTorch operations without custom autograd function"""
    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        if self.training:
            # Generate gate: 1 = identity, 0 = shake
            gate = torch.rand(1, device=x.device) > self.p_drop
            
            if not gate:
                # Forward shake
                alpha_shape = [x.size(0)] + [1] * (x.dim() - 1)
                alpha = torch.empty(alpha_shape, device=x.device).uniform_(*self.alpha_range)
                return x * alpha
            else:
                return x
        else:
            # During evaluation, we apply expected value
            return (1 - self.p_drop) * x
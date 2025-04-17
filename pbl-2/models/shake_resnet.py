"""resnet with ShakeDrop regularization

Based on the paper:
'ShakeDrop Regularization for Deep Residual Learning'
by Yoshihiro Yamada, Masakazu Iwamura, Takuya Akiba, and Koichi Kise
https://arxiv.org/abs/1802.02375
"""

import torch
import torch.nn as nn
from tools.shakedrop import ShakeDrop

class BasicBlockWithShakeDrop(nn.Module):
    """Basic Block for resnet 18 and resnet 34 with ShakeDrop regularization"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, p_drop=0.5, alpha_range=[-1, 1]):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlockWithShakeDrop.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlockWithShakeDrop.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # if dimensions don't match, use 1x1 convolution to match dimensions
        if stride != 1 or in_channels != BasicBlockWithShakeDrop.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlockWithShakeDrop.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlockWithShakeDrop.expansion)
            )
        
        # Add ShakeDrop regularization
        self.shake_drop = ShakeDrop(p_drop=p_drop, alpha_range=alpha_range)

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        
        # Apply ShakeDrop to the residual branch
        residual = self.shake_drop(residual)
        
        # Add the shortcut (skip connection) to the residual branch
        output = residual + shortcut
        
        # Apply ReLU
        output = nn.functional.relu(output)
        
        return output


class BottleNeckWithShakeDrop(nn.Module):
    """Bottleneck Block for resnet over 50 layers with ShakeDrop regularization"""
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, p_drop=0.5, alpha_range=[-1, 1]):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeckWithShakeDrop.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeckWithShakeDrop.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeckWithShakeDrop.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeckWithShakeDrop.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeckWithShakeDrop.expansion)
            )
            
        # Add ShakeDrop regularization
        self.shake_drop = ShakeDrop(p_drop=p_drop, alpha_range=alpha_range)

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        
        # Apply ShakeDrop to the residual branch
        residual = self.shake_drop(residual)
        
        # Add the shortcut (skip connection) to the residual branch
        output = residual + shortcut
        
        # Apply ReLU
        output = nn.functional.relu(output)
        
        return output


class ShakeResNet(nn.Module):
    """ResNet with ShakeDrop regularization"""

    def __init__(self, block, num_block, num_classes=100, p_drop_base=0.5, alpha_range=[-1, 1]):
        super().__init__()

        self.in_channels = 64
        
        # Store total number of blocks for p_drop calculation
        total_blocks = sum(num_block)
        self.total_blocks = total_blocks
        self.p_drop_base = p_drop_base
        self.alpha_range = alpha_range
        self.block_count = 0  # Counter for linear decay rule

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
            
        # Build the ResNet layers with ShakeDrop
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a layer of residual blocks with ShakeDrop regularization"""
        
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            # Calculate p_drop using linear decay rule from the paper
            # The deeper the block, the smaller p_drop becomes
            p_l = 1.0 - (self.block_count / self.total_blocks) * (1.0 - self.p_drop_base)
            
            # Create block with ShakeDrop
            layers.append(block(
                self.in_channels, 
                out_channels, 
                stride, 
                p_drop=p_l,
                alpha_range=self.alpha_range
            ))
            
            self.in_channels = out_channels * block.expansion
            self.block_count += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def shake_resnet18(num_classes=100, p_drop=0.5, alpha_range=[-1, 1]):
    """Return a ShakeResNet-18 model"""
    return ShakeResNet(BasicBlockWithShakeDrop, [2, 2, 2, 2], 
                       num_classes=num_classes, 
                       p_drop_base=p_drop, 
                       alpha_range=alpha_range)


def shake_resnet34(num_classes=100, p_drop=0.5, alpha_range=[-1, 1]):
    """Return a ShakeResNet-34 model"""
    return ShakeResNet(BasicBlockWithShakeDrop, [3, 4, 6, 3], 
                       num_classes=num_classes, 
                       p_drop_base=p_drop, 
                       alpha_range=alpha_range)


def shake_resnet50(num_classes=100, p_drop=0.5, alpha_range=[-1, 1]):
    """Return a ShakeResNet-50 model"""
    return ShakeResNet(BottleNeckWithShakeDrop, [3, 4, 6, 3], 
                       num_classes=num_classes, 
                       p_drop_base=p_drop, 
                       alpha_range=alpha_range)


def shake_resnet101(num_classes=100, p_drop=0.5, alpha_range=[-1, 1]):
    """Return a ShakeResNet-101 model"""
    return ShakeResNet(BottleNeckWithShakeDrop, [3, 4, 23, 3], 
                       num_classes=num_classes, 
                       p_drop_base=p_drop, 
                       alpha_range=alpha_range)


def shake_resnet152(num_classes=100, p_drop=0.5, alpha_range=[-1, 1]):
    """Return a ShakeResNet-152 model"""
    return ShakeResNet(BottleNeckWithShakeDrop, [3, 8, 36, 3], 
                       num_classes=num_classes, 
                       p_drop_base=p_drop, 
                       alpha_range=alpha_range)
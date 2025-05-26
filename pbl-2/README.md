# ** CIFAR100-Classification **
### Pytorch-CIFAR100 
Advanced practice on CIFAR-100 using PyTorch 

### Requirements
This is our experiment environment
1. python3.11
2. pytorch 2.x+cu121
3. wandb 0.19.9
4. NVIDIA RTX A5000 (2 GPUs)



### Restrictions
- 50,000 training data
- No outsource
- No test time adaptation
- Training time (~24 hours per model)
- Can use multiple GPUs (we used 2)

## Table of Usage
- [Dataset](#Dataset)
- [Data Preprocessing](#Data-preprocessing)
- [Data Augmentation](#Data-augmentation)
- [Optimizer](#Optimizer)
- [Run wandb(optional)](#Run-wandb-optional)
- [Train the model](#Train-the-model)
- [Results](#Results)
- [Our Best Model](#Our-best-model)
- [Utility](#Utility)
- [Others](#Others)

## Dataset
We conducted a project to classify images using the CIFAR-100 dataset.    

## Data-preprocessing
- Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
- CIFAR100_TRAIN_MEAN = (0.5071, 0.4867, 0.4408)
- CIFAR100_TRAIN_STD = (0.2675, 0.2565, 0.2761)

## Data-augmentation
- RandomCrop(32, padding=4)
- RandomHorizontalFlip()
- CutMix (alpha=1.0, probability=0.5)

## Optimizer
We used **SAM (Sharpness Aware Minimization)** optimizer which seeks parameters that lie in neighborhoods having uniformly low loss. This improves model generalization.
- Base optimizer: SGD
- SAM specific parameters:
  - rho: 0.05 (ResNet18) / default (ShakePyramidNet)
  - adaptive: False

## Train-the-model
You can use three main files:
- **resnet18-cfc,lr=0.1,factor=0.5,SAM_SGD.ipynb** <- Model 1 (ensemble weight: 0.3)
- **shake_pyramidnet-SAM_SGD.ipynb** <- Model 2 (ensemble weight: 0.7)
- **ensemble.ipynb** <- **For ensemble evaluation**

All models use test data for per-epoch evaluation (no separate validation set).

## Results
| Model                  | Scheduler            | Optimizer | Augmentation | Training Samples | Epochs | Best Epoch | Top 1 Acc | Top 5 Acc | Runtime |
|------------------------|----------------------|-----------|--------------|------------------|--------|------------|-----------|-----------|---------|
| **ResNet18**          | ReduceLROnPlateau    | SAM-SGD   | CutMix       | 50000           | 300    | 210         | **80.36** | **91.39** | ~8h 11m 10s    |
| **ShakeDrop + PyramidNet**   | ReduceLROnPlateau    | SAM-SGD   | CutMix       | 50000           | 300    | 89         | **81.17** | **96.31** | ~22 15m 5s    |
| **Ensemble (0.3:0.7)**| -                    | -         | -            | -               | -      | -          | **82.92** | **96.82** | ~1h 10m     |


## Our-best-model
We conducted model ensemble by combining **ShakePyramidNet** (depth=110, alpha=270) with **ResNet18**, both trained with SAM optimizer.

### Parameters of ResNet18:
```python
{
    "model": "resnet18",
    "batch_size": 128,
    "num_epochs": 300,
    "learning_rate": 0.005,
    "optimizer": "SAM_SGD",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    
    # SAM optimizer settings
    "rho": 0.05,
    "adaptive": False,
    
    # Training settings
    "seed": 2025,
    "patience": 30,
    "cutmix_alpha": 1.0,
    "cutmix_prob": 0.5,
    "warmup_epochs": 10,
    
    # Scheduler settings
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_factor": 0.5,
    "lr_patience": 5,
    "min_lr": 1e-6
}
```

### Parameters of ShakePyramidNet:
```python
{
    "model": "shake_pyramidnet",
    "batch_size": 256,
    "num_epochs": 300,
    "learning_rate": 0.1,
    "optimizer": "SAM_SGD",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    
    # Model architecture
    "depth": 110,
    "alpha": 270,
    
    # Training settings
    "seed": 2025,
    "patience": 30,
    "cutmix_alpha": 1.0,
    "cutmix_prob": 0.5,
    "warmup_epochs": 5,
    
    # Scheduler settings
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_factor": 0.5,
    "lr_patience": 5
}
```

### Ensemble Results:
Different weight combinations were tested:

| ResNet18 Weight | ShakePyramidNet Weight | Top-1 Accuracy | Top-5 Accuracy |
|-----------------|------------------------|----------------|----------------|
| 0.30            | 0.70                   | **82.92**      | **96.82**      |
| 0.40            | 0.60                   | 82.89          | 96.85          |
| 0.45            | 0.55                   | 82.86          | 96.85          |
| 0.50            | 0.50                   | 82.64          | 96.78          |
| 0.00            | 1.00                   | 82.25          | 96.82          |
| 1.00            | 0.00                   | 80.33          | 95.50          |

**Best ensemble configuration: 30% ResNet18 + 70% ShakePyramidNet**

## Utility
The project implements several utilities in `tools/tool.py`:
- **AccuracyEarlyStopping**: Stops training when accuracy stops improving
- **WarmUpLR**: Gradual learning rate warm-up scheduler
- **SAM**: Sharpness Aware Minimization optimizer implementation

## Key Features
1. **SAM Optimizer**: Improves generalization by finding flat minima
2. **CutMix Augmentation**: Advanced data augmentation technique
3. **Warm-up Training**: Gradual learning rate increase for stable training
4. **Ensemble Learning**: Combines predictions from multiple models
5. **Multi-GPU Support**: Utilizes DataParallel for faster training

## Others
- All models use CrossEntropyLoss as the loss function
- Early stopping based on test accuracy with patience of 30 epochs
- Learning rate reduction on plateau with patience of 5 epochs
- Models are saved when achieving best test accuracy
- CUDA memory optimization with `pin_memory=True`

### Performance Improvements
Compared to baseline models:
- **ResNet18 improvement**: +2.56% over single ResNet18
- **ShakePyramidNet improvement**: +0.92% over single ShakePyramidNet

### Notes
- The ensemble evaluation encountered CUDNN warnings but completed successfully
- Each model was trained independently before ensemble evaluation
- Soft voting (weighted average of probabilities) was used for ensemble


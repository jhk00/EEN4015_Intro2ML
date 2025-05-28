# ** PBL - 3 Office Home Classification **
### Multi-task learning on Office-Home dataset using PyTorch 


### Requirements
This is my experiment environment
1. python 3.11
2. pytorch 2.x+cu121
3. wandb 0.19.9
4. torchmetrics
5. timm
6. NVIDIA GPUs (Multi-GPU supported)


### Restrictions
- Office-Home dataset (4 domains, 65 classes)
- Training time (~24 hours per model)
- Used multiple GPUs


## Table of Contents
- [Dataset](#Dataset)
- [Model Architecture](#Model-Architecture)
- [Knowledge Distillation](#Knowledge-Distillation)
- [Data Preprocessing](#Data-preprocessing)
- [Data Augmentation](#Data-augmentation)
- [Optimizer](#Optimizer)
- [Train the model](#Train-the-model)
- [Results](#Results)
- [Our Best Model](#Our-best-model)
- [Utility](#Utility)
- [Key Features](#Key-Features)


## Dataset
This project performs multi-task classification using the Office-Home dataset:
- **4 domains**: Art, Clipart, Product, Real World
- **65 object categories** per domain
- **15,500 images** in total
- **Task 1**: Domain classification (4 classes)
- **Task 2**: Category classification (65 classes)

## Model Architecture


### Student Model: Dilated ResNet50
- **Backbone**: ResNet50 with dilated convolutions (dilate_scale=8)
- **Multi-task branches**:
  - Domain branch at Layer3 output (1024 channels → 128 → 64 → 4)
  - Class branch at Layer4 output (2048 channels → 256 → 128 → 65)
- **Frozen backbone** with trainable task-specific branches


### Teacher Model: EfficientNet-B2
- **Backbone**: EfficientNet-B2 Noisy Student (pretrained)
- **Multi-task heads**:
  - Domain head: 1408 → 512 → 128 → 4
  - Class head: 1408 → 1024 → 512 → 65


## Knowledge Distillation
Knowledge distillation from EfficientNet teacher to ResNet student:
- **Domain KD settings**:
  - α (KD weight): 0.8
  - Temperature: 5.0
- **Class KD settings**:
  - α (KD weight): 0.6
  - Temperature: 3.0
- **Task weighting**: Domain 0.2, Class 0.8


## Data-preprocessing
- Resize to (255, 255) for training, (224, 224) for testing
- Normalize with ImageNet statistics:
  - Mean: (0.485, 0.456, 0.406)
  - Std: (0.229, 0.224, 0.225)


## Data-augmentation
- RandomCrop(224, 224) from 255×255 images
- RandomHorizontalFlip()


## Optimizer
- **Optimizer**: Adam
  - Learning rate: 0.0001 (differential rates for different components)
- **Scheduler**: ReduceLROnPlateau
  - Mode: max (based on mAP)
  - Factor: 0.1
  - Patience: 5 epochs


## Train-the-model
Main training files:
- **main.py** - Original training script
- **paste.txt** - Resume training script with best model loading


Training process:
1. Pre-train teacher model (20 epochs) 
2. Train student with knowledge distillation
3. Use DualMAP early stopping (monitors both domain and class mAP)


## Results
| Model | Teacher | Method | Epochs | Best Epoch | Domain mAP | Class mAP | Runtime |
|-------|---------|--------|--------|------------|------------|-----------|---------|
| **Dilated ResNet50** | EfficientNet-B2 | KD | 58/300 | ~32 | **86.45%** | **87.11%** | ~3h 10m |


### Training Progress
- Learning rate reduced at epoch ~32
- Early stopping triggered at epoch 58


## Our-best-model
The best model uses Knowledge Distillation to transfer knowledge from a larger EfficientNet-B2 teacher to a more efficient Dilated ResNet50 student.


### Key Parameters:
```python
{
    # Model architecture
    "model": "resnet50",
    "teacher_model": "efficientnet_b2_ns",
    "dilate_scale": 8,
    
    # Training settings
    "batch_size": 128,
    "num_epochs": 300,
    "learning_rate": 0.0001,
    "optimizer": "Adam",
    "seed": 2025,
    
    # Multi-task settings
    "num_domains": 4,
    "num_classes": 65,
    "domain_weight": 0.2,
    "class_weight": 0.8,
    
    # Knowledge Distillation
    "teacher_epochs": 20,
    "domain_kd_alpha": 0.8,
    "class_kd_alpha": 0.6,
    "domain_temperature": 5.0,
    "class_temperature": 3.0,
    
    # Early stopping
    "patience": 20,
    "save_every": 5,
    
    # Scheduler
    "scheduler": "ReduceLROnPlateau",
    "scheduler_mode": "max",
    "scheduler_factor": 0.1,
    "scheduler_patience": 5
}
```

### Model Performance:
- **Domain Classification**: 86.45% mAP
- **Class Classification**: 87.11% mAP
- **Convergence**: Achieved best performance around epoch 32

## Utility
The project implements several utilities in `utility/utils.py`:
- **DualMAPEarlyStopping**: Monitors both domain and class mAP for early stopping


## Key Features
1. **Multi-Task Learning**: Simultaneous domain and class classification
2. **Knowledge Distillation**: Efficient knowledge transfer from large teacher to compact student
3. **Dilated Convolutions**: Increased receptive field without losing resolution
4. **Frozen Backbone**: Only task-specific layers are trained for efficiency


### Architecture Advantages
- **Efficiency**: Frozen backbone reduces training time and parameters
- **Multi-scale Features**: Domain uses mid-level features, class uses high-level features


### Notes
- Student training emphasizes class classification (0.8) over domain (0.2)


### Future Improvements
- Try different backbone architectures
- Experiment with different KD techniques ( Self-Knowledge Distillation )


### Reference


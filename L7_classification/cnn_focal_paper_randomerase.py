import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import numpy as np
import wandb  # Import WandB library

# WandB API key setup and initialization
wandb.login(key="ef091b9abcea3186341ddf8995d62bde62d7469e")
wandb.init(project="mnist-cnn", name="combined-augmentation-random-erasing")  # Project and experiment name

# Record hyperparameters
config = {
    "num_epochs": 50,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": 10,
    "focal_loss_gamma": 2.0,  # Focal Loss gamma parameter
    "focal_loss_alpha": 1.0,   # Focal Loss alpha parameter
    "augmentation_probability": 0.7,  # Data augmentation probability
    "random_erasing_probability": 0.5,  # Random Erasing probability
    "random_erasing_scale": (0.02, 0.2),  # Size ratio of area to erase
    "random_erasing_ratio": (0.3, 3.3),  # Aspect ratio of erased area
    "random_erasing_value": 0  # Value to fill erased area (0: black)
}
wandb.config.update(config)  # Record hyperparameters to WandB

# Print version information
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# Check CUDA status
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Check CUDA version
    try:
        print(f"CUDA version: {torch.version.cuda}")
    except:
        print("Unable to check CUDA version.")

# Force GPU setup
try:
    device = torch.device('cuda:0')
    _ = torch.zeros(1).to(device)  # GPU test
    print("GPU is available.")
except:
    print("GPU not available, running on CPU.")
    device = torch.device('cpu')

print(f"Using device: {device}")

num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
num_classes = config["num_classes"]  # 0~9

# Define custom augmentation transform classes compatible with PIL Image
class BackslashTransform:
    def __init__(self, factor=0.3):
        self.factor = factor
        
    def __call__(self, img):
        # Process PIL Image
        return transforms.functional.affine(
            img, 
            angle=0, 
            translate=(0, 0), 
            scale=1.0, 
            shear=(self.factor * 45, 0)  # Backslash direction slope
        )

class SlashTransform:
    def __init__(self, factor=0.3):
        self.factor = factor
        
    def __call__(self, img):
        # Process PIL Image
        return transforms.functional.affine(
            img, 
            angle=0, 
            translate=(0, 0), 
            scale=1.0, 
            shear=(0, self.factor * 45)  # Slash direction slope
        )

class MultiplicationTransform:
    def __init__(self, factor=0.2):
        self.factor = factor
        self.backslash = BackslashTransform(factor=self.factor)
        self.slash = SlashTransform(factor=self.factor)
        
    def __call__(self, img):
        # Apply backslash first then slash
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
            shear=(0, self.factor * 45)  # Right direction slope for '>' shape
        )

class OTransform:
    def __init__(self, distortion=0.3):
        self.distortion = distortion
        
    def __call__(self, img):
        # Get PIL Image size
        width, height = img.size
        
        # 'O' shape distortion approximated with perspective transform
        startpoints = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]  # Original image corners
        endpoints = [
            (int(0 + width * self.distortion), int(0 + height * self.distortion)),  # Top left
            (int(width - 1 - width * self.distortion), int(0 + height * self.distortion)),  # Top right 
            (int(0 + width * self.distortion), int(height - 1 - height * self.distortion)),  # Bottom left
            (int(width - 1 - width * self.distortion), int(height - 1 - height * self.distortion))  # Bottom right
        ]
        
        return transforms.functional.perspective(img, startpoints, endpoints, fill=0)

# Combined transformation class that applies multiple augmentations sequentially
class CombinedAugmentation:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list
    
    def __call__(self, img):
        for transform in self.transforms_list:
            img = transform(img)
        return img

# Create PIL Image-based transform objects
backslash_transform = BackslashTransform(factor=0.3)
slash_transform = SlashTransform(factor=0.3)
greater_than_transform = GreaterThanTransform(factor=0.25)
o_transform = OTransform(distortion=0.3)
rotation_transform = transforms.RandomRotation(degrees=15)
affine_transform = transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.9, 1.1), shear=10)
perspective_transform = transforms.RandomPerspective(distortion_scale=0.2, p=1.0)

# Random Erasing transform
random_erasing_transform = transforms.RandomErasing(
    p=config["random_erasing_probability"],
    scale=config["random_erasing_scale"],
    ratio=config["random_erasing_ratio"],
    value=config["random_erasing_value"]
)

# Various dual augmentation combinations like in the paper
combined_augmentations = [
    # Combinations mentioned in the paper
    CombinedAugmentation([backslash_transform, slash_transform]),  # Combination 1: \ and / combined
    CombinedAugmentation([backslash_transform, greater_than_transform]),  # Combination 2: \ and > combined
    CombinedAugmentation([slash_transform, o_transform]),  # Combination 3: / and O combined
    
    # Additional experimental combinations
    CombinedAugmentation([rotation_transform, backslash_transform]),  # Combination 4: rotation and \ combined
    CombinedAugmentation([affine_transform, o_transform]),  # Combination 5: affine transform and O combined
    CombinedAugmentation([perspective_transform, slash_transform]),  # Combination 6: perspective transform and / combined
]

# Data preprocessing and augmentation - applying various augmentation techniques
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize image to fit CNN
    transforms.RandomChoice([
        # Original image (no augmentation)
        transforms.Lambda(lambda x: x),
        
        # Single augmentation techniques (about 30% probability)
        backslash_transform, 
        slash_transform, 
        MultiplicationTransform(factor=0.2),
        greater_than_transform,
        o_transform,
        affine_transform,
        
        # Combined augmentation techniques (about 70% probability for 2 or more augmentations)
        *combined_augmentations  # Various combined augmentation techniques
    ]),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean and std
    random_erasing_transform  # Apply Random Erasing after ToTensor
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize image to fit CNN
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
])

# Load datasets
train_dataset = datasets.MNIST(root='./data/', train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN model - 3 convolutional layers and a deeper classifier
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # First Convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Input: 1x32x32, Output: 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32x16x16
        )

        # Second Convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Input: 32x16x16, Output: 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64x8x8
        )

        # Third Convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Input: 64x8x8, Output: 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128x4x4
        )

        # Classifier (Fully Connected layers)
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

# Implement Focal Loss
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

# Initialize model and set device
model = CNN(num_classes=num_classes).to(device)
wandb.watch(model, log="all")  # Set WandB to track model weights and gradients

# Switch to Focal Loss
criterion = FocalLoss(alpha=config["focal_loss_alpha"], gamma=config["focal_loss_gamma"])
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

# Training function
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

# Evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    # Lists to store predictions and actual labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()

            # Predictions and confidence
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return loss_sum / len(data_loader), 100. * correct / total, all_preds, all_labels

# Run training and evaluation
train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_acc = 0.0
best_model_path = 'mnist_combined_aug_random_erasing_best.pth'

print("===== Training Start =====")
for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Evaluate on test set every epoch
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Log confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=test_labels,
            preds=test_preds,
            class_names=[str(i) for i in range(num_classes)]
        ),
        "test_accuracy": test_acc,
        "test_loss": test_loss
    })
    
    # Save model (based on test accuracy)
    save_message = ""
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        save_message = f" - New best accuracy! Model saved"
    
    print(f'Epoch: {epoch+1}/{num_epochs}    Train Loss: {train_loss:.4f}    Train Acc: {train_acc:.2f}%    Test Loss: {test_loss:.4f}    Test Acc: {test_acc:.2f}%{save_message}')

    # Log to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "learning_rate": current_lr
    })

# Final test evaluation
print("\n===== Final Evaluation =====")
# Load best performing model
model.load_state_dict(torch.load(best_model_path))
final_test_loss, final_test_acc, final_test_preds, final_test_labels = evaluate(model, test_loader, criterion, device)
print(f"Final test accuracy: {final_test_acc:.2f}%")

# Log final results to WandB
wandb.log({
    "final_test_accuracy": final_test_acc,
    "final_test_loss": final_test_loss,
    "final_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=final_test_labels,
        preds=final_test_preds,
        class_names=[str(i) for i in range(num_classes)]
    )
})

# Visualize training results
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

# Save graph image to WandB
wandb.log({"learning_curves": wandb.Image(plt)})

plt.savefig('learning_curves_combined_aug_random_erasing.png')
plt.show()

# Save final model
torch.save(model.state_dict(), 'mnist_combined_aug_random_erasing_final.pth')
print("Final model saved as 'mnist_combined_aug_random_erasing_final.pth'")
print(f"Best model saved as '{best_model_path}' (Test accuracy: {best_acc:.2f}%)")

# Augmentation visualization function - Fixed PIL Image related errors
def visualize_augmentations():
    # Get original MNIST samples
    # Create dataset without applying image transforms
    original_dataset = datasets.MNIST(root='./data/', train=True, download=False, transform=None)
    
    # Select sample images for each digit type
    samples = []
    for digit in range(5):  # Show only digits 0-4
        indices = (original_dataset.targets == digit).nonzero().squeeze()
        samples.append(original_dataset[indices[0]][0])  # PIL Image
    
    # For Random Erasing visualization
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    # Define augmentation transforms
    augmentations = [
        ("Original", lambda x: x),
        ("Backslash(\\)", BackslashTransform(factor=0.3)),
        ("Slash(/)", SlashTransform(factor=0.3)),
        ("Multiply(×)", MultiplicationTransform(factor=0.2)),
        ("Greater than(>)", GreaterThanTransform(factor=0.25)),
        ("O shape", OTransform(distortion=0.3)),
        ("\\+/", combined_augmentations[0]),
        ("\\+>", combined_augmentations[1]),
        ("/+O", combined_augmentations[2]),
        # Add Random Erasing sample
        ("Random Erasing", lambda x: to_pil(random_erasing_transform(to_tensor(x))))
    ]
    
    # Visualization
    fig, axes = plt.subplots(len(samples), len(augmentations), figsize=(18, 10))
    
    for i, sample in enumerate(samples):
        for j, (aug_name, aug_transform) in enumerate(augmentations):
            # Copy original image (PIL Image)
            img = sample.copy()
            
            # Apply augmentation
            if j > 0:  # First column is original
                img = aug_transform(img)
            
            # Display image
            if i == 0:  # Show title only on first row
                axes[i, j].set_title(aug_name, fontsize=10)
            
            # Convert PIL Image to numpy array for display
            img_array = np.array(img)
            axes[i, j].imshow(img_array, cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples_with_random_erasing.png', dpi=150)
    
    # Log image to WandB
    wandb.log({"augmentation_examples": wandb.Image('augmentation_examples_with_random_erasing.png')})
    
    plt.show()
    return fig

# Run augmentation visualization
try:
    fig = visualize_augmentations()
    print("Augmentation visualization completed")
except Exception as e:
    print(f"Error during augmentation visualization: {e}")

# Class accuracy analysis
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
    
    # Calculate per-class accuracy
    class_accuracy = [100 * correct / total for correct, total in zip(class_correct, class_total)]
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), class_accuracy, color='skyblue')
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Test Accuracy')
    plt.xticks(range(num_classes))
    plt.ylim(90, 100)  # MNIST typically shows >90% accuracy
    plt.grid(axis='y', alpha=0.3)
    
    for i, acc in enumerate(class_accuracy):
        plt.text(i, acc - 0.5, f'{acc:.1f}%', ha='center', va='top')
    
    plt.savefig('class_accuracy_random_erasing.png')
    
    # Log to WandB
    wandb.log({
        "class_accuracy": wandb.Image('class_accuracy_random_erasing.png'),
        "class_accuracy_values": {f"class_{i}_acc": acc for i, acc in enumerate(class_accuracy)}
    })
    
    plt.show()
    
    return class_accuracy

# Run class accuracy analysis
class_accuracy = analyze_class_accuracy()
print("\nPer-class accuracy:")
for i, acc in enumerate(class_accuracy):
    print(f"Digit {i}: {acc:.2f}%")

# Misclassified image analysis
print("\n===== Misclassified Image Analysis =====")
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

        # Find misclassified images
        misclassified = ~predicted.eq(labels)
        if misclassified.sum() > 0:
            misclassified_indices = misclassified.nonzero().squeeze()
            if misclassified_indices.dim() == 0:  # Single index case
                misclassified_indices = misclassified_indices.unsqueeze(0)

            for idx in misclassified_indices:
                if len(misclassified_images) < 25:  # Store up to 25 images
                    misclassified_images.append(images[idx].cpu())
                    misclassified_info.append({
                        'true': labels[idx].item(),
                        'pred': predicted[idx].item(),
                        'conf': confidence[idx].item()
                    })

# Visualize misclassified images
if len(misclassified_images) > 0:
    n_images = len(misclassified_images)
    rows = int(np.ceil(np.sqrt(n_images)))
    cols = int(np.ceil(n_images / rows))

    plt.figure(figsize=(cols * 2.5, rows * 2.5))

    for i in range(n_images):
        plt.subplot(rows, cols, i + 1)
        img = misclassified_images[i].squeeze().numpy()
        # Reverse normalization
        img = img * 0.3081 + 0.1307
        plt.imshow(img, cmap='gray')
        info = misclassified_info[i]
        plt.title(f"T: {info['true']}, P: {info['pred']}\nConf: {info['conf']:.2f}")
        plt.axis('off')

    # Set save path
    misclassified_path = 'misclassified_examples_random_erasing.png'
    plt.tight_layout()
    plt.savefig(misclassified_path, dpi=150)
    
    # Log image to WandB
    wandb.log({"misclassified_examples": wandb.Image(misclassified_path)})
    
    plt.show()

# Experiment summary
print("\n===== Experiment Summary =====")
print(f"Final test accuracy: {final_test_acc:.2f}%")
print(f"Training epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {config['learning_rate']}")
print(f"Focal Loss gamma: {config['focal_loss_gamma']}")
print(f"Focal Loss alpha: {config['focal_loss_alpha']}")
print(f"Random Erasing probability: {config['random_erasing_probability']}")
print("\nApplied various custom augmentations (using fixed learning rate without scheduler)")

# End WandB session
wandb.finish()
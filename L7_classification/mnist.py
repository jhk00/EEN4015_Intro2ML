import torch
from torchvision import datasets, transforms

num_epochs = 20
batch_size = 128

trasnform = transforms.Compose([
    trasnforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root ='./data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    # train
    for images, labels in train_loader:
        # do something

    
    for images, labels in test_loader:
        # do something



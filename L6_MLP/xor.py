import torch
import torch.nn as nn

epochs = 1000

X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
Y = torch.FloatTensor([[0],[1],[1],[0]]) 

y = torch.tensor([0,1,1,0]).reshape(4,1)

model = nn.Sequential(
          nn.Linear(2, 10, bias=True), # 2,10
          nn.ReLU(),
          nn.Linear(10, 10, bias=True), # 10,10
          nn.ReLU(),
          nn.Linear(10, 10, bias=True), # 10,10
          nn.ReLU(),
          nn.Linear(10, 1, bias=True), # 10,1
          nn.ReLU()
          )

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(epochs):
    forward = model(X)
    
    loss = criterion(forward,Y)
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    if (epoch + 1) % 10 ==0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")


with torch.no_grad():
    predictions = model(X)
    predicted_classes = (predictions >= 0.5).float()
    print(predicted_classes)


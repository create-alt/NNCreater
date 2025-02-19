import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 実行時には下記のimportファイル名およびmodule名を変更してください
from test import test

# モデルのインスタンス化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 実行時には下記のmodule名を変更してください
model = test().to(device)

# データの準備
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 損失関数と最適化手法の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images.view(BATCH_SIZE, -1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    train_accuracy = correct / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.4f}")

# モデルの評価
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images.view(BATCH_SIZE, -1))
        correct += (outputs.argmax(dim=1) == labels).sum().item()

test_accuracy = correct / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

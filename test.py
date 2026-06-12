from pillow_heif import register_heif_opener
register_heif_opener()

import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import os

# 1. 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])
# 2. 데이터 로드 위에 추가
from PIL import Image
import os

# 깨진 이미지 확인 및 제거
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
for root, dirs, files in os.walk(data_path):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            Image.open(filepath).verify()
        except:
            print(f"깨진 이미지 삭제: {filepath}")
            os.remove(filepath)
# 2. 데이터 로드
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
train_data = datasets.ImageFolder(data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

print(f"클래스: {train_data.classes}")
print(f"총 이미지 수: {len(train_data)}")

# 3. ResNet18 불러오기
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)

# 4. 학습
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(5):
    total_loss = 0
    correct = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    
    acc = correct / len(train_data) * 100
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {acc:.1f}%")

# 5. 모델 저장
torch.save(model.state_dict(), 'stair_classifier.pth')
print("모델 저장 완료!")
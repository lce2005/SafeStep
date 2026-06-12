from pillow_heif import register_heif_opener
register_heif_opener()

import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from PIL import Image
import os

# 1. 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# 2. 유효하지 않은 이미지 제거
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

for root, dirs, files in os.walk(data_path):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception as e:
            print(f"스킵 삭제: {filepath} - {e}")
            os.remove(filepath)

# 3. 데이터 로드
train_data = datasets.ImageFolder(data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

print(f"클래스: {train_data.classes}")
print(f"총 이미지 수: {len(train_data)}")

# 4. ResNet18 불러오기
model = models.resnet18(weights='IMAGENET1K_V1')  # deprecated 경고 제거
model.fc = nn.Linear(512, 2)

# 5. 학습
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

# 6. 모델 저장
torch.save(model.state_dict(), 'stair_classifier.pth')
print("모델 저장 완료!")
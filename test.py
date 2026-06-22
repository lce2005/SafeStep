from pillow_heif import register_heif_opener
register_heif_opener()

import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from PIL import Image
import os

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),       # 데이터 증강 (좌우 반전)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 불량이미지제거ㅓ
for root, dirs, files in os.walk(data_path):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            Image.open(filepath).verify()
        except Exception:
            print(f"스킵 삭제: {filepath}")
            os.remove(filepath)

# load data
train_data = datasets.ImageFolder(data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

print(f"클래스: {train_data.classes}")
print(f"총 이미지 수: {len(train_data)}")

# ground가 좀 적은거 보정
counts = [0, 0]
for _, label in train_data.samples:
    counts[label] += 1
print(f"클래스별 개수: ground={counts[0]}, stair={counts[1]}")
weights = torch.tensor([sum(counts)/c for c in counts], dtype=torch.float)

# 분류기
model = models.resnet18(weights='IMAGENET1K_V1')

model.fc = nn.Sequential(
    nn.Linear(512, 256),   # 은닉층 1
    nn.ReLU(),             # 활성화 함수 
    nn.Dropout(0.5),       # 과적합 방지
    nn.Linear(256, 64),    # 은닉층 2
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 2)       # 출력층 
)

# 학습
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(weight=weights)   # 불균형 보정 적용

model.train()
for epoch in range(20):      
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
    print(f"Epoch {epoch+1:2d} | Loss: {total_loss:.4f} | Acc: {acc:.1f}%")

# 저장
torch.save(model.state_dict(), 'stair_classifier.pth')
print("모델 저장 완료!")
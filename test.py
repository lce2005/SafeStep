import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn

# 1. 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# 2. 데이터 로드
# 폴더 구조: data/train/stair/, data/train/ground/
train_data = datasets.ImageFolder('data', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

print(f"클래스: {train_data.classes}")  # ['ground', 'stair'] 확인
print(f"총 이미지 수: {len(train_data)}")

# 3. ResNet18 불러오기
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)  # 2개 클래스 (stair/ground)

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
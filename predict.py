# 계단/평지 예측 파일 
# stair -> True , ground -> False
from pillow_heif import register_heif_opener
register_heif_opener()

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

_device = torch.device("cpu")

_model = models.resnet18(weights=None)   

_model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 2)
)

_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stair_classifier.pth")
_model.load_state_dict(torch.load(_model_path, map_location=_device))
_model.eval()   # 추론 모드 (Dropout 꺼짐)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_classes = ["ground", "stair"]


def predict_image(img):
    """PIL 이미지 또는 파일 경로를 받아 ('ground'/'stair', 확신도) 반환"""
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        out = _model(x)
        prob = torch.softmax(out, dim=1)[0]
        idx = prob.argmax().item()
    return _classes[idx], prob[idx].item()


def is_stair(img):
    """계단이면 True, 평지면 False — MediaPipe 담당이 호출용"""
    label, _ = predict_image(img)
    return label == "stair"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        label, conf = predict_image(path)
        print(f"{path}")
        print(f"  → {label} (확신도 {conf*100:.1f}%)")
    else:
        print("사용법: python predict.py <이미지경로>")
        print("예시:   python predict.py data/stair/1.jpg")

from pillow_heif import register_heif_opener
register_heif_opener()

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# 모델 로드
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
_model.eval()

# 전처리 (학습과 동일, 증강 제외)
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_classes = ["ground", "stair"]


def predict_image(img, is_bgr=True):
    """
    img: 파일경로(str) / PIL이미지 / OpenCV프레임(NumPy) 다 받음
    is_bgr: OpenCV 프레임이면 True (BGR→RGB 변환). PIL/경로면 무시됨.
    """
    if isinstance(img, str):                  # 파일 경로
        img = Image.open(img)
    elif isinstance(img, np.ndarray):         # OpenCV 프레임 (NumPy)
        if is_bgr:                            # OpenCV는 BGR이라 RGB로 뒤집기
            img = img[:, :, ::-1]
        img = Image.fromarray(img)
    img = img.convert("RGB")

    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        out = _model(x)
        prob = torch.softmax(out, dim=1)[0]
        idx = prob.argmax().item()
    return _classes[idx], prob[idx].item()


def is_stair(img, is_bgr=True):
    """계단이면 True, 평지면 False — MediaPipe 담당이 호출용"""
    label, _ = predict_image(img, is_bgr=is_bgr)
    return label == "stair"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        label, conf = predict_image(path)
        print(f"{path}")
        print(f"  -> {label} (확신도 {conf*100:.1f}%)")
    else:
        print("사용법: python predict.py <이미지경로>")

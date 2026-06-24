# SafeStep
### Intelligent Fall Detection & Location-Aware Emergency Response System

**MediaPipe · ResNet18 · TTS/STT · Real-time Vision**

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00897B?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [System-architecture](#system-architecture)
- [Emergency-protocols](#emergency-protocol)
- [Project-structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Tech-stack](tech-stack)
- [theoretical-background](#theoretical-background)
- [License](#license)

## Overview

**SafeStep**은 낙상 사고를 실시간으로 감지하고, 사고 발생 **장소(계단/평지)** 에 따라 차별화된 응급 대응을 자동으로 수행하는 지능형 시스템입니다.

1. **계단 인식 (CNN)** — 현재 카메라에 비치는 장소가 계단인지 평지인지 분류합니다.
2. **낙상 감지 (MediaPipe)** — 계단으로 판별된 경우, 사람의 관절 좌표를 추적하여 낙상 여부를 판정합니다.
3. **응급 상호작용 (STT/TTS)** — 낙상이 확정되면 음성으로 사용자의 상태를 확인하고, 필요시 자동으로 119에 신고합니다.

## System Architecture

```
카메라 (실시간 프레임)
    │
    ▼
┌─────────────────────┐
│   predict.py        │  CNN (stair_classifier.pth)
│   is_stair(frame)   │──── 계단? / 평지? → 응급 프로토콜 분기 결정
└────────┬────────────┘
         │ 낙상 감지 ▼
┌─────────────────────┐
│  fall_detection.py  │  MediaPipe Pose
│  process_fall_      │──── 낙하 속도 / 기울기 / 종횡비
│  detection()        │     → 낙상 판정
└────────┬────────────┘
         │ 낙상 확정 시 ▼
┌─────────────────────┐
│  stt_tts_           │   Google STT + gTTS
│  interaction.py     │──── "괜찮으세요?" → 응답 분석
│  handle_fall()      │     → 신고 / 모니터링 / 종료
└─────────────────────┘
```

## Emergency Protocol

프로그램의 판단 흐름은 아래와 같습니다.

```mermaid
flowchart TD
    A[/실시간 영상 입력/] --> B{계단여부 판별}

    B -->|위험| C[계단 판별]
    B -->|저위험| D[계단 아님]

    C --> E[낙상감지 발동]
    E -->|낙상 감지| F[TTS,STT]
    E -->|낙상 미감지| M[계속 모니터링]

    F --> G{응답}
    F --> H{무응답}

    G -->|괜찮아| I[안전 확인]
    G -->|안괜찮아| J[신고]

    H --> J
    I --> M
    J --> M

    D --> M
    M --> A
```

## Project Structure

```
SafeStep/
├── data/                    # 학습 데이터셋
│   ├── stair/               # 계단 이미지
│   └── ground/              # 평지 이미지
├── main.py                  # 메인 실행 파일 (전체 파이프라인 통합)
├── predict.py               # CNN 기반 계단/평지 분류 추론
├── fall_detection.py        # MediaPipe 기반 낙상 감지 + 움직임 모니터링
├── stt_tts_interaction.py   # STT/TTS 응급 상호작용 및 신고 프로토콜
├── train_model.py           # CNN 모델 학습 스크립트
├── predict_video.py         # 영상 파일 대상 계단 분류 테스트
├── convert.py               # 데이터 전처리/변환 유틸리티
├── stair_classifier.pth     # 학습된 CNN 모델 가중치
├── data/                    # 학습 데이터셋
└── LICENSE                  # MIT License
```

## Requirements

```
Python >= 3.8
```

| 패키지 | 용도 |
|---|---|
| `opencv-python` | 웹캠 영상 입력 및 화면 렌더링 |
| `mediapipe` | 실시간 인체 관절 좌표 추출 (Module 1) |
| `numpy` | 관절 좌표 수치 계산 |
| `torch` | PyTorch 딥러닝 프레임워크 (Module 2) |
| `torchvision` | ResNet18 사전학습 모델 로드 |
| `Pillow` | 이미지 전처리 |
| `pillow-heif` | HEIF/HEIC 확장자 이미지 호환 |
| `SpeechRecognition` | STT — 음성을 텍스트로 변환 (Module 3) |
| `gTTS` | TTS — 텍스트를 음성으로 변환 |
| `pygame` | TTS 음성 파일 재생 |
| `PyAudio` | 마이크 하드웨어 접근 |

```bash
pip install opencv-python mediapipe numpy torch torchvision Pillow pillow-heif SpeechRecognition gTTS pygame PyAudio
```

> **Note**: 음성 인식을 사용하려면 마이크가 연결되어 있어야 하며, Google STT API 사용을 위해 인터넷 연결이 필요합니다.

## Usage

### 1️. Git 설치 확인

터미널을 열고 아래 명령어로 Git이 설치되어 있는지 확인하세요:

```bash
git --version
```

> Git이 없다면 [https://git-scm.com/downloads](https://git-scm.com/downloads) 에서 먼저 설치하세요.

---

### 2️. 프로젝트 다운로드

#### 방법 1: Git Clone (권장)

> **Git Clone이란?**  
> 원격 GitHub 저장소의 모든 파일과 코드를 내 컴퓨터로 복사해오는 명령어입니다.  
> 아래 명령어 한 줄로 프로젝트 전체를 내려받을 수 있습니다.

```bash
# 원하는 폴더로 이동 후 아래 명령어 입력
git clone https://github.com/lce2005/SafeStep.git

# 클론된 폴더로 이동
cd SafeStep
```

#### 방법 2: ZIP 파일 다운로드 (Git 없이 간단하게)

1. [https://github.com/lce2005/SafeStep](https://github.com/lce2005/SafeStep) 접속
2. 초록색 **`<> Code`** 버튼 클릭
3. **`Download ZIP`** 클릭하여 다운로드
4. 다운로드된 ZIP 파일 압축 해제
5. 압축 해제된 `SafeStep` 폴더를 터미널에서 열기

---

### 3️. 가상환경 생성 (권장)

다른 프로젝트와 패키지 충돌을 방지하기 위해 가상환경 사용을 권장합니다.

**Anaconda 사용 시 (권장):**
```bash
conda create -n safestep python=3.8 -y
conda activate safestep
```

**venv 사용 시:**
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

---

### 4️. 패키지 설치

```bash
pip install opencv-python mediapipe numpy torch torchvision Pillow pillow-heif SpeechRecognition gTTS pygame PyAudio
```

> **PyAudio 설치 오류 시 (Windows)**:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```
>
> **PyAudio 설치 오류 시 (macOS)**:
> ```bash
> brew install portaudio
> pip install pyaudio
> ```

---

### 5️. 학습된 모델 가중치 배치

`stair_classifier.pth` 파일을 프로젝트 루트 디렉토리에 위치시킵니다:

```
SafeStep/
├── stair_classifier.pth      ← 여기에 배치
├── main.py
├── fall_detection.py
├── stt_tts_interaction.py
├── train_model.py
├── predict.py
├── predict_video.py
└── ...
```

---

### 6️. 실행

```bash
python main.py
```

웹캠이 자동으로 켜지며 실시간 낙상 감지 및 장소 분류가 시작됩니다.

> **종료**: 실행 중 `esc` 키를 누르면 종료됩니다.

`fall_detection.py`에서 조정할 수 있는 주요 임계값입니다.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `TILT_ANGLE_THRESH` | 50 | 어깨-골반 라인 각도 |
| `ASPECT_THRESH` | 1.5 | 바운딩 박스 세로/가로 비율 |
| `FALL_FRAMES` | 10 | 낙상 연속 판정에 필요한 프레임 수 |

조건이 동시에 충족되고, `FALL_FRAMES` 프레임 연속으로 유지될 때 최종 낙상으로 판정합니다.

## Tech Stack

| 영역 | 기술 |
|---|---|
| 계단 분류 | PyTorch CNN (`stair_classifier.pth`) |
| 낙상 감지 | MediaPipe Pose (관절 좌표 추적) |
| 음성 인식 | Google Speech Recognition API |
| 음성 출력 | Google Text-to-Speech (gTTS) |
| 영상 처리 | OpenCV |

## Theoretical Background

### Module 1: Fall Detection — MediaPipe Pose

**MediaPipe Pose**는 Google이 개발한 실시간 인체 자세 추정 라이브러리로,  
단일 RGB 카메라 영상에서 **33개의 신체 랜드마크(관절 좌표)** 를 실시간으로 추출합니다.

SafeStep은 아래 **3가지 조건을 만족**할 때 낙상으로 판정합니다.

#### ① 골반-무릎 Y좌표 역전 감지

정상 기립 상태에서는 골반(hip)의 Y좌표가 무릎(knee)보다 항상 위에 위치합니다.  
낙상 시 이 관계가 역전됩니다.

$$
\text{Fall Condition 1}: \quad Y_{hip} \geq Y_{knee}
$$

#### ② 신체 중심축 기울기 분석

어깨(shoulder)와 골반(hip)을 잇는 몸의 중심 라인이 지면과 이루는 각도를 계산합니다.  
낙상 시 이 각도가 급격히 감소합니다.

$$
\theta = \arctan\left(\frac{Y_{shoulder} - Y_{hip}}{X_{shoulder} - X_{hip}}\right) \times \frac{180}{\pi}
$$

$$
\text{Fall Condition 2}: \quad \theta \leq 50°
$$

#### ③ 바운딩 박스 종횡비 변화 감지

신체를 감싸는 가상의 바운딩 박스(Bounding Box)의 종횡비(Aspect Ratio)를 실시간으로 추적합니다.  
정상 상태(세로형)에서 낙상 상태(가로형)로 변할 때 비율이 감소합니다.

$$
\text{Aspect Ratio} = \frac{H_{bbox}}{W_{bbox}}
$$

$$
\text{Fall Condition 3}: \quad \text{Aspect Ratio} \leq 1.5
$$

> **최종 판별**: 위 3가지 조건이 **만족**될 때 `Fallen State = True`로 판정합니다.

---

### Module 2: Location Classification — ResNet18 CNN

#### ResNet (Residual Network) 이란?

**ResNet**은 Microsoft Research가 2015년 발표한 딥러닝 모델로,  
**잔차 연결(Residual Connection, Skip Connection)** 을 도입하여  
매우 깊은 신경망에서도 기울기 소실(Vanishing Gradient) 문제를 해결한 혁신적인 CNN 구조입니다.

#### 잔차 블록 (Residual Block)

일반 CNN은 입력 $x$를 레이어에 통과시켜 $F(x)$를 학습합니다.  
ResNet은 여기에 원본 입력 $x$를 더하는 **Skip Connection**을 추가합니다:

$$
\text{Output} = F(x) + x
$$

$$
F(x) = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2
$$

> 여기서 $\sigma$는 ReLU 활성화 함수, $W$는 가중치 행렬, $b$는 편향(bias)입니다.

#### 전이 학습 (Transfer Learning)

SafeStep은 **ImageNet으로 사전 학습된 ResNet18** 모델의 가중치를 가져와,  
마지막 Fully Connected Layer만 **계단/평지 2-class 분류**에 맞게 교체하여 학습합니다.

$$
\text{Output} = \text{Softmax}(W_{fc} \cdot \text{features} + b_{fc}), \quad \text{classes} = \{\text{stairs},\ \text{ground}\}
$$

이 방식은 적은 데이터셋으로도 높은 분류 정확도를 달성할 수 있게 합니다.

---

### Module 3: Voice Interaction — TTS / STT

#### STT (Speech-to-Text)

**Google Speech Recognition API**를 활용하여 마이크로 입력된 사용자의 음성을 텍스트로 변환합니다.  
`SpeechRecognition` 라이브러리의 **동적 주변 노이즈 조절(`adjust_for_ambient_noise`)** 을 적용하여  
소음 환경에서도 인식률을 향상시킵니다.

#### TTS (Text-to-Speech)

**Google TTS(gTTS)** 를 활용하여 상황에 맞는 안내 멘트를 자연스러운 한국어 음성으로 변환하고,  
`pygame`으로 재생합니다.

## License

This project is licensed under the [MIT License](LICENSE).

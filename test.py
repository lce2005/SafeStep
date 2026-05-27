import cv2
import numpy as np
import mediapipe as mp
import time

# 1. 최신 미디어파이프 문법에 맞게 Pose 모듈로 변경 (Holistic의 버그 및 에러 해결)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# 동영상 파일 로드
cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# DIVX 코덱이 먹히지 않을 때를 대비해 가 장 무난한 XVID로 변경
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./pose_result.avi', fourcc, fps, (w, h))

prev_time = time.time()

# 2. 실시간 비디오 처리를 위해 static_image_mode를 False로 설정 (성능 가속)
# 낙상 감지 신뢰도를 위해 model_complexity는 높은 값(2) 유지
with mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("동영상 재생 파일의 끝이거나 소스가 없습니다.")
            break
        
        curr_time = time.time()

        # 미디어파이프 입력 처리를 위해 BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # 결과 캔버스 복사
        annotated_image = image.copy()
        
        # 3. 신체 포즈 관절(33개) 정보 추출 및 시각화
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # 💡 [채은님 파트 핵심] 여기서 실시간 랜드마크 좌표 데이터를 뽑아볼 수 있습니다.
            # 예시: 엉덩이/어깨 좌표가 찍히는지 테스트하고 싶다면 아래 주석을 풀어보세요.
            # landmarks = results.pose_landmarks.landmark
            # print(f"Nose Y: {landmarks.y}") 

        # FPS 연산 로직 (제로 디비전 에러 방지)
        sec = curr_time - prev_time
        prev_time = curr_time
        fps_val = 1 / sec if sec > 0 else 0
        fps_str = f"FPS : {fps_val:.1f}"

        # 화면에 FPS 텍스트 출력
        cv2.putText(annotated_image, fps_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 비디오 파일 저장 및 화면 출력
        out.write(annotated_image)
        cv2.imshow('MediaPipe Pose Result', annotated_image)
        
        # ESC 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
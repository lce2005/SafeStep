# fall_detector.py
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from predict import is_stair 

FALL_VEL_THRESH = 0.28   # 초당 y 낙하 속도
TILT_ANGLE_THRESH = 60    # 어깨-골반 라인 각도(누울수록 0)
ASPECT_THRESH = 1.3       # 세로/가로 비율
HIST_LEN = 5              # 속도 계산용 히스토리
FALL_FRAMES = 30          # 낙상 연속 판정 프레임 수

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

y_hist = deque(maxlen=HIST_LEN)
fall_counter = 0

def get_angle_deg(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = np.degrees(np.arctan2(dy, dx))
    return abs(90 - abs(ang))

def process_fall_detection(frame, fps=30.0):
    """
    프레임(frame)을 넣으면 내부에서 계단 여부를 먼저 판별하고, 
    낙상 로직을 거친 뒤 최종 낙상 여부(True/False)와 시각화된 프레임을 반환합니다.
    """
    global fall_counter, y_hist
    
    h, w, _ = frame.shape

    # 1) 계단 여부 판정
    stair_detected = is_stair(frame)

    is_falling_now = False
    status_text = "OK"
    color = (0, 255, 0)

    # 2) 낙상 감지 (계단일 때만 수행)
    if stair_detected:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            nose = (lm[0].x, lm[0].y)
            l_sh = (lm[11].x, lm[11].y); r_sh = (lm[12].x, lm[12].y)
            l_hip = (lm[23].x, lm[23].y); r_hip = (lm[24].x, lm[24].y)
            l_knee = (lm[25].x, lm[25].y); r_knee = (lm[26].x, lm[26].y)
            l_ank = (lm[27].x, lm[27].y); r_ank = (lm[28].x, lm[28].y)

            # 중심 y
            y_center = np.mean([
                nose[1], l_sh[1], r_sh[1], l_hip[1], r_hip[1],
                l_knee[1], r_knee[1], l_ank[1], r_ank[1]
            ])
            y_hist.append(y_center)

            vel = 0
            if len(y_hist) >= 2:
                vel = (y_hist[-1] - y_hist[-2]) * fps

            mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
            mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
            tilt = get_angle_deg(mid_sh, mid_hip)

            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x_min, x_max = max(min(xs), 0), min(max(xs), 1)
            y_min, y_max = max(min(ys), 0), min(max(ys), 1)
            bw = (x_max - x_min) * w
            bh = (y_max - y_min) * h
            aspect = bh / (bw + 1e-6)

            fast_drop = vel > FALL_VEL_THRESH
            low_tilt = tilt <= TILT_ANGLE_THRESH
            flat_box = aspect <= ASPECT_THRESH

            is_falling_now = fast_drop and low_tilt and flat_box

            # 시각화 데이터 그리기
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(frame, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), (255, 0, 0), 2)
            cv2.putText(frame, f"vel: {vel:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"tilt: {tilt:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"aspect: {aspect:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # 평지 → 계단 진입 시 속도 오차 방지
        y_hist.clear()

    # 3) 지속 프레임 기반 최종 낙상 판단
    if is_falling_now:
        fall_counter += 1
    else:
        if fall_counter < FALL_FRAMES:
            fall_counter = max(0, fall_counter - 1)

    fall_flag = fall_counter >= FALL_FRAMES

    # 상태 표시 텍스트 결정
    if fall_flag:
        status_text = "FALL"
        color = (0, 0, 255)
    elif stair_detected:
        status_text = "STAIR (SCANNING)"
        color = (255, 191, 0)
    else:
        status_text = "NOT STAIR (IDLE)"
        color = (0, 255, 0)

    # UI 텍스트 출력
    cv2.putText(frame, status_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, f"counter: {fall_counter}/{FALL_FRAMES}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return fall_flag, frame
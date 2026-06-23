import cv2
import mediapipe as mp
import numpy as np
import time  # 시간 측정을 위해 추가
from collections import deque

from predict import is_stair 

# --- 직관적인 관절 상태 기준 임계값 ---
TILT_ANGLE_THRESH = 50    # 어깨-골반 라인 각도 (50도 이하면 상당히 기운 상태)
ASPECT_THRESH = 1.5       # 세로/가로 비율 (1.2 이하면 몸이 가로로 많이 퍼진 상태)
FALL_FRAMES = 10          # 해당 붕괴 자세가 10프레임(약 0.33초) 유지되면 낙상 확정

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 속도 관련 변수 완전 삭제
fall_counter = 0
shared_cap = None

def set_camera(cap_object):
    global shared_cap
    shared_cap = cap_object

def get_angle_deg(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = np.degrees(np.arctan2(dy, dx))
    return abs(90 - abs(ang))

def process_fall_detection(frame, fps=30.0):
    global fall_counter
    
    h, w, _ = frame.shape
    stair_detected = is_stair(frame)

    status_text = "OK"
    color = (0, 255, 0)
    fall_flag = False

    if stair_detected:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            l_sh = (lm[11].x, lm[11].y); r_sh = (lm[12].x, lm[12].y)
            l_hip = (lm[23].x, lm[23].y); r_hip = (lm[24].x, lm[24].y)
            l_knee = (lm[25].x, lm[25].y); r_knee = (lm[26].x, lm[26].y)

            # 1. 관절 높이(Y) 중앙값 계산 (Y는 아래로 갈수록 커짐)
            mid_sh_y = (l_sh[1] + r_sh[1]) / 2
            mid_hip_y = (l_hip[1] + r_hip[1]) / 2
            mid_knee_y = (l_knee[1] + r_knee[1]) / 2

            # [조건 A] 골반이 무릎과 높이가 비슷해지거나 더 아래로 내려갔는가?
            hip_below_knee = mid_hip_y >= mid_knee_y 

            # [조건 B] 몸통이 바닥을 향해 많이 기울었는가?
            mid_sh = ((l_sh[0] + r_sh[0]) / 2, mid_sh_y)
            mid_hip = ((l_hip[0] + r_hip[0]) / 2, mid_hip_y)
            tilt = get_angle_deg(mid_sh, mid_hip)
            low_tilt = tilt <= TILT_ANGLE_THRESH

            # [조건 C] 몸 전체의 실루엣이 가로로 퍼졌는가? (바운딩 박스)
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x_min, x_max = max(min(xs), 0), min(max(xs), 1)
            y_min, y_max = max(min(ys), 0), min(max(ys), 1)
            bw = (x_max - x_min) * w
            bh = (y_max - y_min) * h
            aspect = bh / (bw + 1e-6)
            flat_box = aspect <= ASPECT_THRESH

            # [최종 판단] 몸이 가로로 퍼진 상태에서, 눕거나 엉덩방아를 찧었다면 낙상 상태로 간주
            is_fallen_state = flat_box and (hip_below_knee or low_tilt)

            if is_fallen_state:
                fall_counter += 1
            else:
                fall_counter = max(0, fall_counter - 1)

            # 화면에 관절 데이터 모니터링 (디버깅용)
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(frame, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), (255, 0, 0), 2)
            cv2.putText(frame, f"Aspect: {aspect:.2f} (<{ASPECT_THRESH})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Tilt: {tilt:.1f} (<{TILT_ANGLE_THRESH})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Hip>Knee: {hip_below_knee}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Fallen State: {is_fallen_state}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
    else:
        fall_counter = 0

    fall_flag = fall_counter >= FALL_FRAMES

    if fall_flag:
        fall_counter = 0  # 감지 후 리셋 (중복 트리거 방지)

    if fall_flag:
        status_text = "FALL"
        color = (0, 0, 255)
    elif stair_detected:
        status_text = "STAIR (SCANNING)"
        color = (255, 191, 0)
    else:
        status_text = "NOT STAIR (IDLE)"
        color = (0, 255, 0)

    cv2.putText(frame, status_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, f"counter: {fall_counter}/{FALL_FRAMES}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return fall_flag, frame

def monitor_movement(duration_seconds: int) -> bool:
    global shared_cap
    if shared_cap is None:
        print("[경고] 카메라 객체가 설정되지 않았습니다. 메인 루프 시작 전 set_camera(cap)를 호출해야 합니다.")
        return False

    print(f"[모니터링 가동] {duration_seconds}초 동안 움직임 여부를 감지합니다...")
    start_time = time.time()
    
    # 첫 프레임 기준점 잡기 및 노이즈 제거
    ret, frame1 = shared_cap.read()
    if not ret: return False
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    
    MOTION_PIXEL_THRESH = 8000  # 움직임 판단 임계값
    
    while time.time() - start_time < duration_seconds:
        ret, frame2 = shared_cap.read()
        if not ret: break
            
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
        
        # 프레임 차이 계산 및 이진화
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_count = cv2.countNonZero(thresh)
        
        # 움직임이 임계값을 넘으면 즉시 True 반환 후 종료
        if motion_count > MOTION_PIXEL_THRESH:
            print(f"[움직임 확인] 변화 크기: {motion_count}. 안전 구역 진입으로 판단하여 경보를 해제합니다.")
            return True
            
        gray1 = gray2
        
        # 화면 피드백 유지
        remain_time = duration_seconds - int(time.time() - start_time)
        cv2.putText(frame2, f"Monitoring Movement... {remain_time}s left", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("SafeStep Integrated System", frame2)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC 누르면 비상 탈출
            break
            
    print(f"[모니터링 종료] {duration_seconds}초 동안 움직임이 전혀 없습니다. 최종 119 신고를 접수합니다.")
    return False
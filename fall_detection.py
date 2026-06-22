import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# 설정값
FALL_VEL_THRESH = 0.28   # y축 낙하 속도 임계
TILT_ANGLE_THRESH = 60   # 어깨-엉덩이 라인 기울기 임계 (deg)
ASPECT_THRESH = 1.3      # 세로/가로 비율 임계
HIST_LEN = 5             # 속도 계산용 히스토리 프레임 길이

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_angle_deg(p1, p2):
    # p1, p2: (x, y)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = np.degrees(np.arctan2(dy, dx))
    # 어깨-엉덩이 라인을 세로 기준으로 보기 위해 90도 보정
    return abs(90 - abs(ang))

def main():
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)#카메라에서 FPS를 가져옴
    if fps <= 1e-3:#드라이버/카메라가 FPS를 못 가져오는 경우
        fps = 30.0

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    y_hist = deque(maxlen=HIST_LEN) 

    fall_counter = 0 # 낙상 감지 카운터
    fall_flag = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        # 이번 프레임에서 최종 조건을 충족했는지 판별할 변수 초기화
        is_falling_now = False

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # 관심 관절: 코(0), 양쪽 어깨(11,12), 양쪽 엉덩이(23,24)
            nose = (lm[0].x, lm[0].y)
            l_sh = (lm[11].x, lm[11].y)
            r_sh = (lm[12].x, lm[12].y)
            l_hip = (lm[23].x, lm[23].y)
            r_hip = (lm[24].x, lm[24].y)
            l_knee = (lm[25].x, lm[25].y) #무릎 관절 추가
            r_knee = (lm[26].x, lm[26].y)
            l_ank = (lm[27].x, lm[27].y) #발목 관절 추가
            r_ank = (lm[28].x, lm[28].y)

            # 중심 y 평균
            y_center = np.mean([
                nose[1],
                l_sh[1], r_sh[1],
                l_hip[1], r_hip[1],
                l_knee[1], r_knee[1],
                l_ank[1], r_ank[1],
            ])
            y_hist.append(y_center)

            vel = 0
            if len(y_hist) >= 2:
                vel = (y_hist[-1] - y_hist[-2]) * fps  # 초당 y 변화량(아래 +)
            # 기울기: 양쪽 어깨 중간과 엉덩이 중간
            mid_sh = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
            mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
            tilt = get_angle_deg(mid_sh, mid_hip)  # 0에 가까울수록 누움

            # 바운딩 박스 (관절 전체)
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x_min, x_max = max(min(xs), 0), min(max(xs), 1)
            y_min, y_max = max(min(ys), 0), min(max(ys), 1)
            bw = (x_max - x_min) * w
            bh = (y_max - y_min) * h
            aspect = bh / (bw + 1e-6)  # 세로/가로

            # 조건
            fast_drop = vel > FALL_VEL_THRESH
            low_tilt = tilt <= TILT_ANGLE_THRESH
            flat_box = aspect <= ASPECT_THRESH

            is_falling_now = fast_drop and low_tilt and flat_box#현재 프레임에서 낙상 조건 충족 여부

            # 시각화
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 가상의 박스 그리기
            cv2.rectangle(
                frame,
                (int(x_min * w), int(y_min * h)),
                (int(x_max * w), int(y_max * h)),
                (255, 0, 0), 2
            )
            
            cv2.putText(frame, f"vel: {vel:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"tilt: {tilt:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"aspect: {aspect:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        

        #지속 시간 체크 로직 연동
        if is_falling_now:
            fall_counter += 1
        else:
            # 이미 30프레임 이상 누워있어서 확정 FALL 상태가 되었다면, 
            # 중간에 노이즈로 잠깐 부르르 떨어도 FALL 상태를 유지하기 위해 30 미만일 때만 리셋
            if fall_counter < 30: 
                fall_counter = max(0, fall_counter - 1)
        fall_flag = fall_counter >= 30  # 30프레임 이상 조건 충족 시 낙상 확정


        status = "FALL" if fall_flag else "OK"
        color = (0,0,255) if fall_flag else (0,255,0)

        cv2.putText(frame, status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, f"counter: {fall_counter}/30", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("SafeStep Fall Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty("SafeStep Fall Detection", cv2.WND_PROP_VISIBLE) < 1:
            break#창이 닫히면 루프 종료

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

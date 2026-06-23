import cv2
from predict import is_stair
from fall_detection import process_fall_detection, monitor_movement, set_camera
from stt_tts_interaction import EmergencyInteraction

def main():
    # 카메라 전원 켜기
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3: 
        fps = 30.0

    # 카메라 영상 쉐어링
    set_camera(cap)

    # 응급 대화 인터랙션 객체 세팅
    interaction = EmergencyInteraction(movement_monitor=monitor_movement)

    print("SafeStep 시스템이 가동되었습니다. (종료하려면 ESC 키를 누르세요)")

    while True:
        # 카메라에서 실시간으로 프레임 읽어오기
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 불러올 수 없습니다.")
            break

        # [핵심 1] 낙상 감지 모듈 가동 (이 안에서 계단 여부도 함께 판별됨)
        # 결과값: is_fall(낙상 여부 True/False), drawn_frame(뼈대가 그려진 화면)
        is_fall, drawn_frame = process_fall_detection(frame, fps)

        # 화면 송출
        cv2.imshow("SafeStep Integrated System", drawn_frame)

        # [핵심 2] 낙상이 확정(True)되었을 때 응급 프로토콜 가동
        if is_fall:
            print("\n[긴급] 낙상이 감지되었습니다! STT/TTS 응급 프로토콜을 시작합니다.")

            # 현재 위치가 계단인지 평지인지 한 번 더 정확히 확인
            is_danger = is_stair(frame)

            # 시스템 가동 (음성으로 묻고, 대답 듣고, 모니터링까지 알아서 진행됨)
            result = interaction.handle_fall(is_danger_area=is_danger)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 루프 탈출
            break

    # 시스템 종료 시 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("시스템이 종료되었습니다.")

if __name__ == "__main__":
    main()
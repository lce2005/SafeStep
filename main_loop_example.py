from stt_tts_interaction import EmergencyInteraction, PrintSpeaker, ConsoleSpeechListener


def detect_stair_area(frame) -> bool:
    """
    1번 팀원의 ResNet 계단/평지 인식 결과를 연결할 자리입니다.
    예: return stair_model.predict(frame) == "stairs"
    """
    return True


def detect_fall(frame) -> bool:
    """
    2번 팀원의 MediaPipe 낙상 감지 결과를 연결할 자리입니다.
    예: return fall_detector.update(frame).is_fall
    """
    return True


def main() -> None:
    interaction = EmergencyInteraction(
        speaker=PrintSpeaker(),
        listener=ConsoleSpeechListener(),
    )

    frame = None
    is_fall = detect_fall(frame)

    if not is_fall:
        return

    is_danger_area = detect_stair_area(frame)
    result = interaction.handle_fall(is_danger_area=is_danger_area)

    if result.should_call_emergency:
        print("응급 대응 단계:", result.emergency_stage)
        print("실제 서비스에서는 여기서 119 신고 API, 보호자 문자, 위치 전송 등을 호출합니다.")
    else:
        print("응급 신고 없이 종료:", result.emergency_stage)


if __name__ == "__main__":
    main()

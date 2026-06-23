from __future__ import annotations

import argparse
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - optional runtime dependency
    sr = None

try:
    from gtts import gTTS
except ImportError:  # pragma: no cover - optional runtime dependency
    gTTS = None

try:
    import pygame
except ImportError:  # pragma: no cover - optional runtime dependency
    pygame = None


EMERGENCY_NUMBER = "119"
NO_RESPONSE_MONITOR_SECONDS = 180


@dataclass
class SpeechAnswer:
    text: Optional[str]
    status: str
    error: Optional[str] = None

    @property
    def has_text(self) -> bool:
        return self.status == "text" and bool(self.text)


@dataclass
class InteractionResult:
    should_call_emergency: bool
    emergency_stage: str
    reason: str
    first_answer: Optional[str] = None
    report_answer: Optional[str] = None
    is_danger_area: bool = False


FallProcessor = Callable[[Any, float], Any]
DangerDetector = Callable[[Any], bool]
FrameProvider = Callable[[], Any]
FpsProvider = Callable[[], float]


class PrintSpeaker:
    def speak(self, text: str) -> None:
        print("[TTS]", text)


class GTTSpeaker:
    def __init__(self, lang: str = "ko") -> None:
        self.lang = lang

    def speak(self, text: str) -> None:
        print("[TTS]", text)

        if gTTS is None or pygame is None:
            print("[TTS] gTTS 또는 pygame이 설치되지 않아 음성 대신 텍스트만 출력합니다.")
            return

        filename = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                filename = fp.name

            gTTS(text=text, lang=self.lang).save(filename)
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as exc:
            print("[TTS] 음성 출력 오류:", exc)
        finally:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
            if filename and os.path.exists(filename):
                os.remove(filename)


class ConsoleSpeechListener:
    def listen(self) -> SpeechAnswer:
        text = input("[STT 대신 입력] 사용자 답변: ").strip()
        if not text:
            return SpeechAnswer(text=None, status="no_response")
        return SpeechAnswer(text=text, status="text")


class ConsoleMovementMonitor:
    def __call__(self, duration_seconds: int) -> bool:
        minutes = duration_seconds // 60
        answer = input(
            f"[움직임 감지 시뮬레이션] {minutes}분 안에 움직임이 있었습니까? (y/n): "
        ).strip().lower()
        return answer in {"y", "yes", "예", "네", "응", "있음"}


class GoogleSpeechListener:
    def __init__(
        self,
        language: str = "ko-KR",
        timeout: int = 5,
        phrase_time_limit: int = 5,
        ambient_duration: float = 0.8,
    ) -> None:
        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.ambient_duration = ambient_duration

    def listen(self) -> SpeechAnswer:
        if sr is None:
            return SpeechAnswer(
                text=None,
                status="error",
                error="speech_recognition 패키지가 설치되어 있지 않습니다.",
            )

        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                print("[STT] 주변 소음을 분석합니다.")
                recognizer.adjust_for_ambient_noise(source, duration=self.ambient_duration)
                print("[STT] 답변을 듣고 있습니다.")
                audio = recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

            text = recognizer.recognize_google(audio, language=self.language)
            print("[STT] 인식 결과:", text)
            return SpeechAnswer(text=text, status="text")
        except sr.WaitTimeoutError:
            return SpeechAnswer(text=None, status="no_response")
        except sr.UnknownValueError:
            return SpeechAnswer(text=None, status="unknown")
        except sr.RequestError as exc:
            return SpeechAnswer(text=None, status="error", error=str(exc))
        except OSError as exc:
            return SpeechAnswer(
                text=None,
                status="error",
                error=f"마이크 장치를 사용할 수 없습니다: {exc}",
            )
        except Exception as exc:
            return SpeechAnswer(
                text=None,
                status="error",
                error=f"음성 입력 초기화에 실패했습니다: {exc}",
            )


def load_default_fall_processor() -> FallProcessor:
    from fall_detection import process_fall_detection

    return process_fall_detection


def _read_bool_from_detection_result(result: Any) -> bool:
    if isinstance(result, bool):
        return result

    if isinstance(result, dict):
        for key in (
            "is_fall",
            "fall_detected",
            "fall_detection",
            "fall_flag",
            "is_falling",
            "fall",
        ):
            if key in result:
                return bool(result[key])
        raise ValueError("낙상 감지 결과 dict에 낙상 여부 키가 없습니다.")

    if isinstance(result, (tuple, list)):
        if not result:
            return False
        return _read_bool_from_detection_result(result[0])

    for attr in (
        "is_fall",
        "fall_detected",
        "fall_detection",
        "fall_flag",
        "is_falling",
        "fall",
    ):
        if hasattr(result, attr):
            return bool(getattr(result, attr))

    return bool(result)


def process_fall_frame(
    frame: Any,
    fps: float,
    interaction: "EmergencyInteraction",
    fall_processor: Optional[FallProcessor] = None,
    is_danger_area_override: Optional[bool] = None,
) -> Optional[InteractionResult]:
    processor = fall_processor or load_default_fall_processor()
    fall_result = processor(frame, fps)

    # process_fall_detection returns (fall_flag, stair_detected, frame)
    is_fall = _read_bool_from_detection_result(fall_result)

    if not is_fall:
        return None

    # 3-tuple이면 두 번째 요소가 stair_detected (위험 장소 여부)
    if isinstance(fall_result, (tuple, list)) and len(fall_result) >= 2:
        is_danger_area = bool(fall_result[1])
    else:
        is_danger_area = False

    if is_danger_area_override is not None:
        is_danger_area = is_danger_area_override

    return interaction.handle_fall(is_danger_area=is_danger_area)


class FallProcessorMovementMonitor:
    def __init__(
        self,
        frame_provider: FrameProvider,
        fps_provider: FpsProvider,
        fall_processor: Optional[FallProcessor] = None,
        check_interval_seconds: float = 0.5,
    ) -> None:
        self.frame_provider = frame_provider
        self.fps_provider = fps_provider
        self.fall_processor = fall_processor or load_default_fall_processor()
        self.check_interval_seconds = check_interval_seconds

    def __call__(self, duration_seconds: int) -> bool:
        end_time = time.monotonic() + duration_seconds

        while time.monotonic() < end_time:
            frame = self.frame_provider()
            fps = self.fps_provider()
            fall_result = self.fall_processor(frame, fps)

            if not _read_bool_from_detection_result(fall_result):
                return True

            time.sleep(self.check_interval_seconds)

        return False


def normalize_korean(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def classify_condition(text: Optional[str]) -> str:
    normalized = normalize_korean(text)
    if not normalized:
        return "no_response"

    explicit_ok = [
        "괜찮",
        "괜찬",
        "안다쳤",
        "안아파",
        "멀쩡",
        "문제없",
        "일어났",
        "일어남",
        "바로일어",
    ]
    explicit_not_ok = [
        "안괜찮",
        "괜찮지않",
        "못일어나",
        "못움직",
        "살려",
        "도와",
        "아파",
        "아프",
        "다쳤",
        "통증",
        "피나",
        "119",
        "응급",
        "구급",
    ]

    if "안다쳤" in normalized or "안아파" in normalized:
        return "ok"

    if any(keyword in normalized for keyword in explicit_not_ok):
        return "not_ok"

    if any(keyword in normalized for keyword in explicit_ok):
        return "ok"

    if normalized in {"아니", "아뇨", "아니요", "아녀"}:
        return "not_ok"

    return "unknown"


def classify_report_intent(text: Optional[str]) -> str:
    normalized = normalize_korean(text)
    if not normalized:
        return "no_response"

    hard_no = [
        "신고하지마",
        "신고하지말",
        "부르지마",
        "부르지말",
        "필요없",
        "하지마",
        "하지말",
        "괜찮",
        "됐어",
        "싫어",
        "싫",
    ]
    yes = [
        "응",
        "어",
        "네",
        "예",
        "그래",
        "좋아",
        "빨리",
        "제발",
        "지금",
        "당장",
        "신고",
        "불러",
        "해줘",
        "살려",
        "도와",
    ]
    simple_no = {"아니", "아뇨", "아니요", "아녀", "노"}

    if any(keyword in normalized for keyword in hard_no):
        return "no"

    if any(keyword in normalized for keyword in yes):
        return "yes"

    if normalized in simple_no:
        return "no"

    return "unknown"


class EmergencyInteraction:
    def __init__(
        self,
        speaker: Optional[object] = None,
        listener: Optional[object] = None,
        emergency_callback: Optional[Callable[[InteractionResult], None]] = None,
        movement_monitor: Optional[Callable[[int], bool]] = None,
        no_response_monitor_seconds: int = NO_RESPONSE_MONITOR_SECONDS,
        retry_on_unknown: int = 1,
    ) -> None:
        self.speaker = speaker or GTTSpeaker()
        self.listener = listener or GoogleSpeechListener()
        self.emergency_callback = emergency_callback or self._default_emergency_callback
        self.movement_monitor = movement_monitor
        self.no_response_monitor_seconds = no_response_monitor_seconds
        self.retry_on_unknown = retry_on_unknown

    def handle_fall(self, is_danger_area: bool) -> InteractionResult:
        self._say("낙상이 감지되었습니다. 괜찮으세요?")

        first_answer = self._listen_with_retry(
            retry_prompt="답변을 이해하지 못했습니다. 괜찮으시면 괜찮다고 말씀해 주세요.",
        )

        if first_answer.status == "error":
            self._say("음성 인식 장치를 사용할 수 없습니다. 마이크 설정을 확인해 주세요.")
            return InteractionResult(
                should_call_emergency=False,
                emergency_stage="STT_ERROR",
                reason=first_answer.error or "음성 인식 오류",
                is_danger_area=is_danger_area,
            )

        condition = classify_condition(first_answer.text)

        if first_answer.status == "no_response" or condition == "no_response":
            return self._handle_no_response(is_danger_area, first_answer)

        if condition == "ok":
            self._say("다행입니다. 혹시 통증이 있으면 병원 방문을 권장드립니다.")
            return InteractionResult(
                should_call_emergency=False,
                emergency_stage="ADVICE_ONLY",
                reason="사용자가 괜찮다고 응답함",
                first_answer=first_answer.text,
                is_danger_area=is_danger_area,
            )

        if condition in {"not_ok", "unknown"}:
            return self._ask_report(first_answer, is_danger_area)

        return self._handle_no_response(is_danger_area, first_answer)

    def _ask_report(
        self,
        first_answer: SpeechAnswer,
        is_danger_area: bool,
    ) -> InteractionResult:
        self._say("대신 119에 신고해 드릴까요?")

        report_answer = self._listen_with_retry(
            retry_prompt="신고를 원하시면 네, 원하지 않으시면 아니요라고 말씀해 주세요.",
        )

        if report_answer.status == "error":
            self._say("음성 인식 장치를 사용할 수 없습니다. 주변 사람에게 도움을 요청해 주세요.")
            return InteractionResult(
                should_call_emergency=False,
                emergency_stage="STT_ERROR",
                reason=report_answer.error or "신고 의사 확인 중 음성 인식 오류",
                first_answer=first_answer.text,
                is_danger_area=is_danger_area,
            )

        intent = classify_report_intent(report_answer.text)

        if intent == "no_response":
            return self._handle_no_response(
                is_danger_area,
                first_answer,
                report_answer=report_answer,
            )

        if intent == "yes":
            result = InteractionResult(
                should_call_emergency=True,
                emergency_stage="USER_REQUESTED_REPORT",
                reason="사용자가 신고를 요청함",
                first_answer=first_answer.text,
                report_answer=report_answer.text,
                is_danger_area=is_danger_area,
            )
            self._report(result)
            return result

        if intent == "no":
            self._say("알겠습니다. 통증이 계속되면 병원 방문을 권장드립니다.")
            return InteractionResult(
                should_call_emergency=False,
                emergency_stage="USER_DECLINED_REPORT",
                reason="사용자가 신고를 거절함",
                first_answer=first_answer.text,
                report_answer=report_answer.text,
                is_danger_area=is_danger_area,
            )

        if is_danger_area:
            result = InteractionResult(
                should_call_emergency=True,
                emergency_stage="AUTO_REPORT_DANGER_AREA",
                reason="신고 의사를 확인하지 못했고 위험 장소로 판단됨",
                first_answer=first_answer.text,
                report_answer=report_answer.text,
                is_danger_area=is_danger_area,
            )
            self._report(result)
            return result

        self._say("응답을 정확히 확인하지 못했습니다. 통증이 있으면 주변에 도움을 요청해 주세요.")
        return InteractionResult(
            should_call_emergency=False,
            emergency_stage="UNCLEAR_RESPONSE_SAFE_AREA",
            reason="신고 의사를 확인하지 못했지만 비위험 장소로 판단됨",
            first_answer=first_answer.text,
            report_answer=report_answer.text,
            is_danger_area=is_danger_area,
        )

    def _handle_no_response(
        self,
        is_danger_area: bool,
        first_answer: SpeechAnswer,
        report_answer: Optional[SpeechAnswer] = None,
    ) -> InteractionResult:
        if is_danger_area:
            self._say("위험 장소에서 응답이 확인되지 않아 즉시 신고합니다.")
            result = InteractionResult(
                should_call_emergency=True,
                emergency_stage="AUTO_REPORT_NO_RESPONSE_DANGER_AREA",
                reason="위험 장소에서 무응답으로 판단되어 즉시 신고함",
                first_answer=first_answer.text,
                report_answer=report_answer.text if report_answer else None,
                is_danger_area=is_danger_area,
            )
            self._report(result)
            return result

        self._say("응답이 확인되지 않았습니다. 3분 동안 움직임을 확인하겠습니다.")

        if self.movement_monitor is None:
            self._say("움직임 감지 모듈이 연결되지 않았습니다.")
            return InteractionResult(
                should_call_emergency=False,
                emergency_stage="MOVEMENT_MONITOR_UNAVAILABLE",
                reason="무응답 후 움직임 감지 모듈이 연결되지 않음",
                first_answer=first_answer.text,
                report_answer=report_answer.text if report_answer else None,
                is_danger_area=is_danger_area,
            )

        movement_detected = self.movement_monitor(self.no_response_monitor_seconds)

        if movement_detected:
            self._say("움직임이 확인되어 신고 절차를 종료합니다.")
            return InteractionResult(
                should_call_emergency=False,
                emergency_stage="NO_RESPONSE_MOVEMENT_DETECTED",
                reason="무응답 후 3분 이내 움직임이 감지됨",
                first_answer=first_answer.text,
                report_answer=report_answer.text if report_answer else None,
                is_danger_area=is_danger_area,
            )

        result = InteractionResult(
            should_call_emergency=True,
            emergency_stage="AUTO_REPORT_NO_MOVEMENT_3_MINUTES",
            reason="무응답 후 3분 동안 움직임이 감지되지 않음",
            first_answer=first_answer.text,
            report_answer=report_answer.text if report_answer else None,
            is_danger_area=is_danger_area,
        )
        self._report(result)
        return result

    def _listen_with_retry(self, retry_prompt: str) -> SpeechAnswer:
        answer = self.listener.listen()

        for _ in range(self.retry_on_unknown):
            if answer.status in {"text", "no_response"}:
                return answer
            if answer.status == "error":
                print("[STT] 오류:", answer.error)
                return answer

            self._say(retry_prompt)
            answer = self.listener.listen()

        return answer

    def _say(self, text: str) -> None:
        self.speaker.speak(text)

    def _report(self, result: InteractionResult) -> None:
        self._say("신고하였습니다.")
        self.emergency_callback(result)

    @staticmethod
    def _default_emergency_callback(result: InteractionResult) -> None:
        print(
            "[EMERGENCY]",
            f"{EMERGENCY_NUMBER} 신고 처리 필요:",
            result.emergency_stage,
            "-",
            result.reason,
        )


def build_interaction(mode: str) -> EmergencyInteraction:
    if mode == "console":
        return EmergencyInteraction(
            speaker=PrintSpeaker(),
            listener=ConsoleSpeechListener(),
            movement_monitor=ConsoleMovementMonitor(),
        )

    from fall_detection import monitor_movement

    return EmergencyInteraction(movement_monitor=monitor_movement)


def main() -> None:
    parser = argparse.ArgumentParser(description="낙상 감지 후 STT/TTS 응급 대화 테스트")
    parser.add_argument(
        "--mode",
        choices=["console", "voice"],
        default="console",
        help="console은 키보드 입력, voice는 실제 마이크와 TTS를 사용합니다.",
    )
    parser.add_argument(
        "--danger",
        action="store_true",
        help="계단 등 위험 장소로 판단된 상황을 테스트합니다.",
    )
    args = parser.parse_args()

    interaction = build_interaction(args.mode)
    result = interaction.handle_fall(is_danger_area=args.danger)

    print("\n[RESULT]")
    print("응급 신고 여부:", result.should_call_emergency)
    print("응급 단계:", result.emergency_stage)
    print("판단 이유:", result.reason)


if __name__ == "__main__":
    main()

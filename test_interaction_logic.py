import unittest

from stt_tts_interaction import (
    EmergencyInteraction,
    InteractionResult,
    SpeechAnswer,
    classify_condition,
    classify_report_intent,
)


class SilentSpeaker:
    def __init__(self):
        self.messages = []

    def speak(self, text):
        self.messages.append(text)


class SequenceListener:
    def __init__(self, answers):
        self.answers = list(answers)

    def listen(self):
        if not self.answers:
            return SpeechAnswer(text=None, status="no_response")
        return self.answers.pop(0)


class InteractionLogicTest(unittest.TestCase):
    def run_flow(self, answers, is_danger_area):
        speaker = SilentSpeaker()
        reported = []

        def on_report(result: InteractionResult):
            reported.append(result)

        interaction = EmergencyInteraction(
            speaker=speaker,
            listener=SequenceListener(answers),
            emergency_callback=on_report,
        )
        result = interaction.handle_fall(is_danger_area=is_danger_area)
        return result, speaker.messages, reported

    def test_condition_keywords(self):
        self.assertEqual(classify_condition("괜찮아"), "ok")
        self.assertEqual(classify_condition("안 다쳤어"), "ok")
        self.assertEqual(classify_condition("넘어졌는데 바로 일어남"), "ok")
        self.assertEqual(classify_condition("안 괜찮아"), "not_ok")
        self.assertEqual(classify_condition("아파 도와줘"), "not_ok")
        self.assertEqual(classify_condition("아니요"), "not_ok")

    def test_report_keywords(self):
        self.assertEqual(classify_report_intent("응 빨리 신고해줘"), "yes")
        self.assertEqual(classify_report_intent("제발 지금 당장"), "yes")
        self.assertEqual(classify_report_intent("아니 신고하지마"), "no")
        self.assertEqual(classify_report_intent("괜찮아 필요 없어"), "no")

    def test_ok_answer_ends_with_advice(self):
        result, messages, reported = self.run_flow(
            answers=[SpeechAnswer(text="괜찮아", status="text")],
            is_danger_area=True,
        )

        self.assertFalse(result.should_call_emergency)
        self.assertEqual(result.emergency_stage, "ADVICE_ONLY")
        self.assertFalse(reported)
        self.assertIn("병원 방문", messages[-1])

    def test_not_ok_and_yes_reports(self):
        result, _, reported = self.run_flow(
            answers=[
                SpeechAnswer(text="아파", status="text"),
                SpeechAnswer(text="응 빨리", status="text"),
            ],
            is_danger_area=False,
        )

        self.assertTrue(result.should_call_emergency)
        self.assertEqual(result.emergency_stage, "USER_REQUESTED_REPORT")
        self.assertEqual(len(reported), 1)

    def test_not_ok_and_no_ends(self):
        result, _, reported = self.run_flow(
            answers=[
                SpeechAnswer(text="도와줘", status="text"),
                SpeechAnswer(text="아니 신고하지마", status="text"),
            ],
            is_danger_area=True,
        )

        self.assertFalse(result.should_call_emergency)
        self.assertEqual(result.emergency_stage, "USER_DECLINED_REPORT")
        self.assertFalse(reported)

    def test_no_response_in_danger_area_reports(self):
        result, _, reported = self.run_flow(
            answers=[SpeechAnswer(text=None, status="no_response")],
            is_danger_area=True,
        )

        self.assertTrue(result.should_call_emergency)
        self.assertEqual(result.emergency_stage, "AUTO_REPORT_NO_RESPONSE_DANGER_AREA")
        self.assertEqual(len(reported), 1)

    def test_no_response_in_safe_area_ends(self):
        result, _, reported = self.run_flow(
            answers=[SpeechAnswer(text=None, status="no_response")],
            is_danger_area=False,
        )

        self.assertFalse(result.should_call_emergency)
        self.assertEqual(result.emergency_stage, "NO_RESPONSE_SAFE_AREA")
        self.assertFalse(reported)


if __name__ == "__main__":
    unittest.main()

import unittest

from verbatim.transcript.words import Utterance
from verbatim.verbatim import filter_merged_non_speech_markers


class TestMergedNonSpeechMarkers(unittest.TestCase):
    def test_overlapping_silence_marker_is_removed(self):
        marker = Utterance.marker("m1", 0, 16000, "[SILENCE]")
        speech = Utterance("u1", "HOST", 8000, 24000, "hello", [])

        result = filter_merged_non_speech_markers([marker, speech])

        self.assertEqual([speech], result)

    def test_nearby_environment_marker_is_removed_within_collar(self):
        marker = Utterance.marker("m1", 0, 16000, "[ENVIRONMENT NOISE]")
        speech = Utterance("u1", "GUEST", 20000, 32000, "bonjour", [])

        result = filter_merged_non_speech_markers([marker, speech])

        self.assertEqual([speech], result)

    def test_marker_without_nearby_speech_is_preserved(self):
        marker = Utterance.marker("m1", 0, 16000, "[SILENCE]")
        speech = Utterance("u1", "HOST", 40000, 56000, "hello", [])

        result = filter_merged_non_speech_markers([marker, speech])

        self.assertEqual([marker, speech], result)

    def test_multiple_markers_without_speech_are_preserved(self):
        marker1 = Utterance.marker("m1", 0, 16000, "[SILENCE]")
        marker2 = Utterance.marker("m2", 24000, 40000, "[ENVIRONMENT NOISE]")

        result = filter_merged_non_speech_markers([marker1, marker2])

        self.assertEqual([marker1, marker2], result)

    def test_null_speaker_dialog_does_not_suppress_marker(self):
        marker = Utterance.marker("m1", 0, 16000, "[SILENCE]")
        null_speaker_dialog = Utterance("u1", None, 8000, 24000, "hello", [])

        result = filter_merged_non_speech_markers([marker, null_speaker_dialog])

        self.assertEqual([marker, null_speaker_dialog], result)


if __name__ == "__main__":
    unittest.main()

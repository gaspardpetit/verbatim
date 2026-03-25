import io
import json
import unittest

from verbatim.transcript.words import Utterance
from verbatim_files.format.json import TranscriptFormatter as JsonTranscriptFormatter
from verbatim_files.format.txt import COLORSCHEME_NONE
from verbatim_files.format.txt import TranscriptFormatter as TextTranscriptFormatter
from verbatim_files.format.writer import (
    LanguageStyle,
    ProbabilityStyle,
    SpeakerStyle,
    TimestampStyle,
)


class TestTranscriptMarkers(unittest.TestCase):
    def test_json_formatter_supports_marker_utterance(self):
        formatter = JsonTranscriptFormatter()
        marker = Utterance.marker("utt1", 0, 96000, "[SILENCE]")

        output = io.BytesIO()
        output.write(formatter.start())
        output.write(formatter.format_utterance(marker))
        output.write(formatter.finish())
        output.seek(0)

        data = json.load(io.TextIOWrapper(output, encoding="utf-8"))
        utterance = data["utterances"][0]
        self.assertEqual("[SILENCE]", utterance["text"])
        self.assertIsNone(utterance["speaker"])
        self.assertIsNone(utterance["language"])
        self.assertEqual([], utterance["words"])

    def test_text_formatter_supports_marker_utterance_without_speaker_placeholder(self):
        formatter = TextTranscriptFormatter(
            speaker_style=SpeakerStyle.always,
            timestamp_style=TimestampStyle.range,
            probability_style=ProbabilityStyle.none,
            language_style=LanguageStyle.always,
        )
        marker = Utterance.marker("utt1", 0, 96000, "[ENVIRONMENT NOISE]")

        rendered = formatter.format_utterance(marker, COLORSCHEME_NONE).decode("utf-8")

        self.assertIn("[00:00:00-00:00:06]:[ENVIRONMENT NOISE]\n", rendered)
        self.assertNotIn("[SPEAKER", rendered)


if __name__ == "__main__":
    unittest.main()

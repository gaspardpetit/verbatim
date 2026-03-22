import unittest
from types import SimpleNamespace

from verbatim_diarization.pyannote.output import select_speaker_diarization


class TestPyannoteOutputSelection(unittest.TestCase):
    def test_prefers_exclusive_speaker_diarization(self):
        regular = object()
        exclusive = object()
        output = SimpleNamespace(
            speaker_diarization=regular,
            exclusive_speaker_diarization=exclusive,
        )

        self.assertIs(select_speaker_diarization(output), exclusive)

    def test_falls_back_to_regular_speaker_diarization(self):
        regular = object()
        output = SimpleNamespace(
            speaker_diarization=regular,
            exclusive_speaker_diarization=None,
        )

        self.assertIs(select_speaker_diarization(output), regular)

    def test_returns_output_when_no_wrapped_diarization_exists(self):
        output = object()

        self.assertIs(select_speaker_diarization(output), output)


if __name__ == "__main__":
    unittest.main()

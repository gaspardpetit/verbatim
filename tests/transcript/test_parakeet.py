"""Parakeet word timestamp integration without loading a native checkpoint."""
# pylint: disable=protected-access

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from verbatim.voices.transcribe.parakeet import ParakeetTranscriber


class TestParakeetTranscriber(unittest.TestCase):
    def test_word_timestamps_use_window_offset_and_preserve_punctuation(self):
        model = Mock()
        model.transcribe.return_value = [
            SimpleNamespace(
                timestamp={
                    "word": [
                        {"start": 0.1, "end": 0.4, "word": "Hello"},
                        {"start": 0.5, "end": 0.8, "word": "world."},
                    ]
                }
            )
        ]
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber._model = model

        words = transcriber.transcribe(
            audio=np.zeros(16000, dtype=np.float32),
            lang=None,
            prompt="",
            prefix="",
            window_ts=32000,
            audio_ts=48000,
        )

        self.assertEqual("Hello world.", "".join(word.word for word in words))
        self.assertEqual([(33600, 38400), (40000, 44800)], [(word.start_ts, word.end_ts) for word in words])
        self.assertEqual(["und", "und"], [word.lang for word in words])
        model.transcribe.assert_called_once()
        self.assertEqual(16000, len(model.transcribe.call_args.args[0][0]))
        self.assertTrue(model.transcribe.call_args.kwargs["timestamps"])
        self.assertFalse(model.transcribe.call_args.kwargs["verbose"])

    def test_rejects_non_16khz_audio_configuration(self):
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber._model = Mock()

        with patch("verbatim.voices.transcribe.parakeet.AUDIO_PARAMS", SimpleNamespace(sample_rate=8000)):
            with self.assertRaisesRegex(RuntimeError, "requires a 16000 Hz Verbatim sample rate"):
                transcriber.transcribe(
                    audio=np.zeros(8000, dtype=np.float32),
                    lang=None,
                    prompt="",
                    prefix="",
                    window_ts=0,
                    audio_ts=8000,
                )

        transcriber._model.transcribe.assert_not_called()

    def test_capabilities_reject_language_hints(self):
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        self.assertFalse(transcriber.get_capabilities().accepts_language_hint)
        with self.assertRaisesRegex(RuntimeError, "does not accept language hints"):
            transcriber.transcribe(
                audio=np.zeros(16000, dtype=np.float32),
                lang="en",
                prompt="",
                prefix="",
                window_ts=0,
                audio_ts=16000,
            )


if __name__ == "__main__":
    unittest.main()

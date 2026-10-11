"""Capability contract tests, including backends with optional native imports."""
# pylint: disable=import-outside-toplevel

import importlib.util
import sys
import unittest
from dataclasses import FrozenInstanceError

from verbatim.voices.transcribe.faster_whisper import FasterWhisperTranscriber
from verbatim.voices.transcribe.qwen_asr import QwenAsrTranscriber
from verbatim.voices.transcribe.voxtral import VoxtralTranscriber
from verbatim_transcript import TranscriberCapabilities


class TestTranscriberCapabilities(unittest.TestCase):
    def test_existing_backends_expose_their_language_and_context_contracts(self):
        expected = {
            FasterWhisperTranscriber: (True, True, False, True, True),
            QwenAsrTranscriber: (True, True, False, True, True),
            VoxtralTranscriber: (True, False, False, False, False),
        }
        mlx_available = all(importlib.util.find_spec(name) is not None for name in ("mlx", "mlx_whisper"))
        if sys.platform != "darwin" or mlx_available:
            from verbatim.voices.transcribe.whispermlx import WhisperMlxTranscriber

            expected[WhisperMlxTranscriber] = (True, True, False, True, False)
        for backend, flags in expected.items():
            with self.subTest(backend=backend.__name__):
                capabilities = backend.__new__(backend).get_capabilities()
                self.assertIsInstance(capabilities, TranscriberCapabilities)
                self.assertEqual(30.0, capabilities.window_duration)
                self.assertEqual(
                    flags,
                    (
                        capabilities.accepts_language_hint,
                        capabilities.detects_language,
                        capabilities.reports_language,
                        capabilities.supports_prompt,
                        capabilities.supports_prefix,
                    ),
                )

    def test_capabilities_cannot_be_mutated(self):
        capabilities = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber).get_capabilities()
        with self.assertRaises(FrozenInstanceError):
            setattr(capabilities, "accepts_language_hint", False)
        with self.assertRaises(FrozenInstanceError):
            setattr(capabilities, "window_duration", 60.0)


if __name__ == "__main__":
    unittest.main()

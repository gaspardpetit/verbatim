import unittest

import numpy as np

from verbatim.config import Config
from verbatim.verbatim import Verbatim


class DummyModels:
    pass


class TestSkipLeadingSilence(unittest.TestCase):
    def test_no_voice_segments_respects_max_skip(self):
        config = Config(device="cpu")
        verbatim = Verbatim(
            config=config,
            models=DummyModels(),
            vad_callback=lambda audio, min_speech_duration_ms, min_silence_duration_ms: [],
        )
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = 32000
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        result = verbatim.skip_leading_silence(max_skip=8000)

        self.assertEqual(8000, verbatim.state.window_ts)
        self.assertEqual(32000, verbatim.state.audio_ts)
        self.assertEqual(32000, result.next_ts)
        self.assertIsNone(result.marker)

    def test_long_zero_energy_skip_emits_silence_marker(self):
        config = Config(device="cpu")
        verbatim = Verbatim(
            config=config,
            models=DummyModels(),
            vad_callback=lambda audio, min_speech_duration_ms, min_silence_duration_ms: [],
        )
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = 100000
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        result = verbatim.skip_leading_silence(max_skip=100000)

        self.assertIsNotNone(result.marker)
        marker = result.marker
        if marker is None:
            self.fail("Expected a silence marker")
        self.assertEqual("[SILENCE]", marker.text)
        self.assertIsNone(marker.speaker)
        self.assertEqual(0, marker.start_ts)
        self.assertGreaterEqual(marker.end_ts, 80000)

    def test_long_high_energy_skip_emits_noise_marker(self):
        config = Config(device="cpu")
        verbatim = Verbatim(
            config=config,
            models=DummyModels(),
            vad_callback=lambda audio, min_speech_duration_ms, min_silence_duration_ms: [],
        )
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = 100000
        verbatim.state.rolling_window.array = np.full(len(verbatim.state.rolling_window.array), 0.05, dtype=np.float32)

        result = verbatim.skip_leading_silence(max_skip=100000)

        self.assertIsNotNone(result.marker)
        marker = result.marker
        if marker is None:
            self.fail("Expected an environment-noise marker")
        self.assertEqual("[ENVIRONMENT NOISE]", marker.text)
        self.assertIsNone(marker.speaker)


if __name__ == "__main__":
    unittest.main()

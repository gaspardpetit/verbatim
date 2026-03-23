import unittest

import numpy as np

from verbatim.config import Config
from verbatim.verbatim import Verbatim


class DummyModels:
    pass


class TestSkipLeadingSilence(unittest.TestCase):
    def test_no_voice_segments_respects_max_skip(self):
        config = Config(device="cpu")
        verbatim = Verbatim(config=config, models=DummyModels(), vad_callback=lambda audio, min_ms, pad_ms: [])
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = 32000
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        returned_ts = verbatim.skip_leading_silence(max_skip=8000)

        self.assertEqual(8000, verbatim.state.window_ts)
        self.assertEqual(32000, verbatim.state.audio_ts)
        self.assertEqual(32000, returned_ts)


if __name__ == "__main__":
    unittest.main()

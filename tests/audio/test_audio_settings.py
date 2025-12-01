import importlib
import os
import unittest
from unittest.mock import patch

from verbatim_audio import settings


class TestAudioSettings(unittest.TestCase):
    def test_env_override(self):
        with patch.dict(
            os.environ,
            {
                "VERBATIM_SAMPLE_RATE": "8000",
                "VERBATIM_FRAME_SIZE": "80",
                "VERBATIM_MAX_WINDOW_FRAMES": "1000",
            },
            clear=True,
        ):
            importlib.reload(settings)
            params = settings.get_audio_params()
            self.assertEqual(params.sample_rate, 8000)
            self.assertEqual(params.frame_size, 80)
            self.assertEqual(params.max_window_frames, 1000)
            self.assertEqual(params.fps, 100)


if __name__ == "__main__":
    unittest.main()

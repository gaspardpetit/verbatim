import os
import tempfile
import unittest

import numpy as np

# pylint: disable=import-outside-toplevel


class TestStereoDiarization(unittest.TestCase):
    def test_determine_speaker_silence(self):
        from verbatim.voices.diarize.stereo import StereoDiarization

        diarizer = StereoDiarization()
        # pylint: disable=protected-access
        speaker = diarizer._determine_speaker(0.0, 0.0, 0.0, 0.0)
        self.assertEqual(speaker, "UNKNOWN")

    def test_compute_diarization_silence(self):
        import soundfile as sf

        from verbatim.voices.diarize.stereo import StereoDiarization

        sample_rate = 16000
        audio = np.zeros((sample_rate, 2), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name

        try:
            diarizer = StereoDiarization()
            annotation = diarizer.compute_diarization(tmp_path, segment_duration=0.5)
            self.assertEqual(len(annotation), 0)
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

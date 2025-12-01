import importlib.util
import os
import tempfile
import unittest

import numpy as np

pyannote_available = importlib.util.find_spec("pyannote") is not None
soundfile_available = importlib.util.find_spec("soundfile") is not None

# pylint: disable=import-outside-toplevel


class TestEnergyDiarization(unittest.TestCase):
    @unittest.skipUnless(pyannote_available, "pyannote not available")
    def test_determine_speaker_silence(self):
        from verbatim_diarization.stereo.diarize import EnergyDiarization

        diarizer = EnergyDiarization()
        # pylint: disable=protected-access
        speaker = diarizer._determine_speaker(0.0, 0.0, 0.0, 0.0)
        self.assertEqual(speaker, "UNKNOWN")

    @unittest.skipUnless(pyannote_available and soundfile_available, "pyannote/soundfile not available")
    def test_compute_diarization_silence(self):
        import soundfile as sf  # type: ignore

        from verbatim_diarization.stereo.diarize import EnergyDiarization

        sample_rate = 16000
        audio = np.zeros((sample_rate, 2), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name

        try:
            diarizer = EnergyDiarization()
            annotation = diarizer.compute_diarization(tmp_path, segment_duration=0.5)
            self.assertEqual(len(annotation), 0)
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

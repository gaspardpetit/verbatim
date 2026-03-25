import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from verbatim.cache import FileBackedArtifactCache
from verbatim_diarization.pyannote.diarize import PyAnnoteDiarization


class _FakePipeline:
    def __init__(self):
        self.calls = []
        self.instantiated = None
        self.device = None

    def instantiate(self, config):
        self.instantiated = config

    def to(self, device):
        self.device = device

    def __call__(self, file_for_pipeline, hook=None, num_speakers=None):
        self.calls.append(
            {
                "file_for_pipeline": file_for_pipeline,
                "hook": hook,
                "num_speakers": num_speakers,
            }
        )
        return SimpleNamespace(
            uri="sample",
            itertracks=lambda yield_label=True: iter(
                [
                    (SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_0"),
                ]
            ),
        )


class _FakeSoundFileModule:
    @staticmethod
    def read(_buffer):
        audio = np.zeros(16000, dtype=np.float32)
        return audio, 16000


class TestPyannoteWaveformInput(unittest.TestCase):
    def test_compute_diarization_uses_waveform_input_without_torchcodec(self):
        cache = FileBackedArtifactCache(base_dir=".")
        cache.set_bytes("sample.wav", b"fake")

        fake_pipeline = _FakePipeline()
        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.read = _FakeSoundFileModule.read

        diarizer = PyAnnoteDiarization(cache=cache, device="cpu", huggingface_token="")
        diarizer.pipeline = fake_pipeline

        with patch.dict(sys.modules, {"soundfile": fake_soundfile}):
            annotation = diarizer.compute_diarization(file_path="sample.wav", nb_speakers=2)

        self.assertEqual(1, len(annotation.segments))
        self.assertEqual("SPEAKER_0", annotation.segments[0].speaker)
        self.assertEqual(2, fake_pipeline.calls[0]["num_speakers"])
        file_for_pipeline = fake_pipeline.calls[0]["file_for_pipeline"]
        self.assertIn("waveform", file_for_pipeline)
        self.assertIn("sample_rate", file_for_pipeline)
        self.assertEqual(16000, file_for_pipeline["sample_rate"])


if __name__ == "__main__":
    unittest.main()

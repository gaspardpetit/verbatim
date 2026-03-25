import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from verbatim.cache import FileBackedArtifactCache, InMemoryArtifactCache
from verbatim_diarization.diarize.factory import create_diarizer
from verbatim_diarization.senko.diarize import SenkoDiarization


class _FakeDiarizer:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def diarize(self, wav_path, accurate=None, generate_colors=False):
        _FakeDiarizer.calls.append(
            {
                "wav_path": wav_path,
                "accurate": accurate,
                "generate_colors": generate_colors,
                "init_kwargs": self.kwargs,
            }
        )
        return {
            "merged_segments": [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
                {"start": 1.0, "end": 2.5, "speaker": "SPEAKER_02"},
            ],
            "timing_stats": {"total_time": 0.25},
        }


class TestSenkoDiarization(unittest.TestCase):
    def setUp(self):
        _FakeDiarizer.calls = []

    def test_factory_creates_senko_strategy(self):
        diarizer = create_diarizer(strategy="senko", device="mps", cache=InMemoryArtifactCache())
        self.assertIsInstance(diarizer, SenkoDiarization)
        self.assertEqual("coreml", diarizer.device)

    def test_compute_diarization_from_existing_wav(self):
        fake_senko = types.ModuleType("senko")
        fake_senko.Diarizer = _FakeDiarizer

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sample.wav"
            sf.write(str(wav_path), np.zeros(16000, dtype=np.int16), 16000, subtype="PCM_16")
            cache = InMemoryArtifactCache()

            with patch.dict(sys.modules, {"senko": fake_senko}):
                diarizer = SenkoDiarization(cache=cache, device="mps")
                annotation = diarizer.compute_diarization(
                    file_path=str(wav_path),
                    out_rttm_file="sample.rttm",
                    out_vttm_file="sample.vttm",
                )

        self.assertEqual(2, len(annotation.segments))
        self.assertEqual("SPEAKER_01", annotation.segments[0].speaker)
        self.assertEqual("coreml", _FakeDiarizer.calls[0]["init_kwargs"]["device"])
        self.assertFalse(_FakeDiarizer.calls[0]["generate_colors"])
        self.assertNotEqual("", cache.get_text("sample.rttm"))
        self.assertNotEqual("", cache.get_text("sample.vttm"))

    def test_constructor_parses_string_accurate_flag(self):
        fake_senko = types.ModuleType("senko")
        fake_senko.Diarizer = _FakeDiarizer

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sample.wav"
            sf.write(str(wav_path), np.zeros(16000, dtype=np.int16), 16000, subtype="PCM_16")
            cache = InMemoryArtifactCache()

            with patch.dict(sys.modules, {"senko": fake_senko}):
                diarizer = SenkoDiarization(cache=cache, device="mps", accurate="false")
                diarizer.compute_diarization(file_path=str(wav_path))

        self.assertFalse(_FakeDiarizer.calls[0]["accurate"])

    def test_requires_disk_backed_cache_for_materialized_wav(self):
        fake_senko = types.ModuleType("senko")
        fake_senko.Diarizer = _FakeDiarizer
        cache = InMemoryArtifactCache()
        cache.set_bytes("sample.mp3", b"fake")

        with patch.dict(sys.modules, {"senko": fake_senko}):
            diarizer = SenkoDiarization(cache=cache, device="cpu")
            with self.assertRaises(RuntimeError):
                diarizer.compute_diarization(file_path="sample.mp3")

    def test_materializes_wav_with_file_backed_cache(self):
        fake_senko = types.ModuleType("senko")
        fake_senko.Diarizer = _FakeDiarizer

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileBackedArtifactCache(base_dir=tmpdir)
            cache.set_bytes("sample.mp3", b"fake-mp3")

            with patch.dict(sys.modules, {"senko": fake_senko}):
                diarizer = SenkoDiarization(cache=cache, device="cpu")
                with patch("verbatim_diarization.senko.diarize.convert_bytes_to_wav", return_value=str(Path(tmpdir) / "sample-senko.wav")):
                    with patch.object(diarizer, "_is_compatible_wav", return_value=False):
                        annotation = diarizer.compute_diarization(file_path="sample.mp3")

        self.assertEqual(2, len(annotation.segments))
        self.assertTrue(_FakeDiarizer.calls[0]["wav_path"].endswith("sample-senko.wav"))


if __name__ == "__main__":
    unittest.main()

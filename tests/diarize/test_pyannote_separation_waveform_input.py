import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from verbatim.cache import FileBackedArtifactCache
from verbatim_diarization.pyannote import separate as pyannote_separate
from verbatim_diarization.pyannote.separate import PyannoteSpeakerSeparation


class _FakeSources:
    def __init__(self):
        self.data = np.array(
            [
                [0.1, 0.0],
                [0.2, 0.1],
                [0.1, 0.3],
            ],
            dtype=np.float32,
        )


class _FakeDiarization:
    def labels(self):
        return ["SPEAKER_0", "SPEAKER_1"]


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, file_for_pipeline, hook=None, num_speakers=None):
        self.calls.append(
            {
                "file_for_pipeline": file_for_pipeline,
                "hook": hook,
                "num_speakers": num_speakers,
            }
        )
        diarization_output = SimpleNamespace(
            speaker_diarization=_FakeDiarization(),
            exclusive_speaker_diarization=None,
        )
        return diarization_output, _FakeSources()


class _FakeSoundFileModule:
    @staticmethod
    def read(_buffer):
        audio = np.zeros(16000, dtype=np.float32)
        return audio, 16000

    @staticmethod
    def write(buffer, speaker_data, sample_rate, **kwargs):
        _ = speaker_data, sample_rate, kwargs
        buffer.write(b"RIFFfakeWAVE")


class TestPyannoteSeparationWaveformInput(unittest.TestCase):
    @staticmethod
    def _run_separation(separator, *, file_path, out_speaker_wav_prefix, nb_speakers):
        return separator._separate_to_audio_refs(  # pylint: disable=protected-access
            file_path=file_path,
            out_speaker_wav_prefix=out_speaker_wav_prefix,
            nb_speakers=nb_speakers,
            status_hook=None,
        )

    def test_separation_uses_waveform_input_without_torchcodec(self):
        cache = FileBackedArtifactCache(base_dir=".")
        cache.set_bytes("sample.wav", b"fake")

        fake_pipeline = _FakePipeline()
        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.read = _FakeSoundFileModule.read
        fake_soundfile.write = _FakeSoundFileModule.write

        separator = PyannoteSpeakerSeparation.__new__(PyannoteSpeakerSeparation)
        separator.cache = cache
        separator.pipeline = fake_pipeline

        with patch.dict(sys.modules, {"soundfile": fake_soundfile}):
            diarization, audio_refs_meta = self._run_separation(
                separator,
                file_path="sample.wav",
                out_speaker_wav_prefix="speaker",
                nb_speakers=2,
            )

        self.assertEqual(["SPEAKER_0", "SPEAKER_1"], list(diarization.labels()))
        self.assertEqual(2, len(audio_refs_meta))
        self.assertEqual(2, fake_pipeline.calls[0]["num_speakers"])
        file_for_pipeline = fake_pipeline.calls[0]["file_for_pipeline"]
        self.assertIn("waveform", file_for_pipeline)
        self.assertIn("sample_rate", file_for_pipeline)
        self.assertEqual(16000, file_for_pipeline["sample_rate"])

    def test_separation_file_path_fallback_enables_torchcodec(self):
        cache = FileBackedArtifactCache(base_dir=".")
        cache.set_bytes("sample.wav", b"fake")

        fake_pipeline = _FakePipeline()
        separator = PyannoteSpeakerSeparation.__new__(PyannoteSpeakerSeparation)
        separator.cache = cache
        separator.pipeline = fake_pipeline

        with patch.object(separator, "_prepare_pipeline_input", return_value=("sample.wav", None)):
            with patch.object(pyannote_separate, "ensure_torchcodec_audio_decoder") as ensure_decoder:
                diarization, audio_refs_meta = self._run_separation(
                    separator,
                    file_path="sample.wav",
                    out_speaker_wav_prefix="speaker",
                    nb_speakers=2,
                )

        ensure_decoder.assert_called_once_with("pyannote separation")
        self.assertEqual(["SPEAKER_0", "SPEAKER_1"], list(diarization.labels()))
        self.assertEqual(2, len(audio_refs_meta))
        self.assertEqual("sample.wav", fake_pipeline.calls[0]["file_for_pipeline"])


if __name__ == "__main__":
    unittest.main()

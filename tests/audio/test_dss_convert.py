import io
import sys
import types
import unittest
import wave
from typing import cast
from unittest.mock import patch

import numpy as np

from verbatim.cache import InMemoryArtifactCache
from verbatim_audio.convert import convert_bytes_to_wav
from verbatim_audio.sources.factory import create_audio_sources


def _test_credential() -> str:
    return "".join(["12", "34"])


class _FakeDecodedAudio:
    def __init__(self, samples, sample_rate, native_rate):
        self.samples = samples
        self.sample_rate = sample_rate
        self.native_rate = native_rate


class TestDssConversion(unittest.TestCase):
    @staticmethod
    def _build_wav_bytes() -> bytes:
        buffer = io.BytesIO()
        wav_file = cast(wave.Wave_write, wave.open(buffer, "wb"))
        with wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 8)
        return buffer.getvalue()

    def test_convert_bytes_to_wav_uses_pydsscodec_for_ds2(self):
        fake_module = types.ModuleType("pydsscodec")
        fake_module.decode_bytes = lambda _data, password=None: _FakeDecodedAudio(
            samples=[0.0, 0.5, -0.5, 1.25],
            sample_rate=16000,
            native_rate=16000,
        )

        cache = InMemoryArtifactCache()
        with patch.dict(sys.modules, {"pydsscodec": fake_module}):
            output_path = convert_bytes_to_wav(
                input_bytes=b"fake-ds2",
                input_label="sample.ds2",
                working_prefix_no_ext="sample",
                cache=cache,
            )

        self.assertEqual(output_path, "sample.wav")
        wav_bytes = cache.get_bytes(output_path)
        self.assertTrue(wav_bytes)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getframerate(), 16000)
            pcm = wav_file.readframes(wav_file.getnframes())

        self.assertEqual(len(pcm), 8)

    def test_convert_bytes_to_wav_requires_optional_dependency_for_dss(self):
        cache = InMemoryArtifactCache()
        with patch.dict(sys.modules, {"pydsscodec": None}):
            with self.assertRaisesRegex(RuntimeError, "pydsscodec"):
                convert_bytes_to_wav(
                    input_bytes=b"fake-dss",
                    input_label="sample.dss",
                    working_prefix_no_ext="sample",
                    cache=cache,
                )

    def test_create_audio_sources_converts_dss_into_wav_backed_file_source(self):
        cache = InMemoryArtifactCache()
        cache.set_bytes("sample.dss", b"fake-dss")

        def _fake_convert(**_kwargs):
            cache.set_bytes("sample.wav", self._build_wav_bytes())
            return "sample.wav"

        with patch("verbatim_audio.sources.factory.convert_bytes_to_wav", side_effect=_fake_convert):
            sources = create_audio_sources(
                input_source="sample.dss",
                device="cpu",
                cache=cache,
            )

        self.assertEqual(len(sources), 1)
        self.assertEqual(getattr(sources[0], "file_path", None), "sample.wav")

    def test_convert_bytes_to_wav_passes_password_to_pydsscodec(self):
        fake_module = types.ModuleType("pydsscodec")
        calls = []
        test_credential = _test_credential()

        def _decode_bytes(_data, password=None):
            calls.append(password)
            return _FakeDecodedAudio(samples=[0.0, 0.25, -0.25], sample_rate=16000, native_rate=16000)

        fake_module.decode_bytes = _decode_bytes

        cache = InMemoryArtifactCache()
        with patch.dict(sys.modules, {"pydsscodec": fake_module}):
            convert_bytes_to_wav(
                input_bytes=b"fake-ds2",
                input_label="sample.ds2",
                working_prefix_no_ext="sample",
                password=test_credential,
                cache=cache,
            )

        self.assertEqual(calls, [test_credential])

    def test_convert_bytes_to_wav_scales_pcm_sized_float_output(self):
        fake_module = types.ModuleType("pydsscodec")
        fake_module.decode_bytes = lambda _data, password=None: _FakeDecodedAudio(
            samples=[-32768.0, -16384.0, 0.0, 16384.0, 32767.0],
            sample_rate=16000,
            native_rate=16000,
        )

        cache = InMemoryArtifactCache()
        with patch.dict(sys.modules, {"pydsscodec": fake_module}):
            output_path = convert_bytes_to_wav(
                input_bytes=b"fake-ds2",
                input_label="sample.ds2",
                working_prefix_no_ext="sample",
                cache=cache,
            )

        with wave.open(io.BytesIO(cache.get_bytes(output_path)), "rb") as wav_file:
            pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)

        np.testing.assert_array_equal(pcm, np.array([-32767, -16383, 0, 16383, 32766], dtype=np.int16))


if __name__ == "__main__":
    unittest.main()

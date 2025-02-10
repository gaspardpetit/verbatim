import math
import unittest
import numpy as np

# pylint: disable=import-outside-toplevel

class TestAudioProcessing(unittest.TestCase):
    def test_format_audio(self):
        from verbatim.audio.audio import format_audio

        sample_rate = 16000
        audio_mono_float = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        output = format_audio(audio_mono_float, sample_rate)
        np.testing.assert_array_almost_equal(output, audio_mono_float)
        self.assertEqual(output.dtype, np.float32)

        audio_mono_int16 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        output = format_audio(audio_mono_int16, sample_rate)
        expected = audio_mono_int16.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(output, expected, decimal=5)
        self.assertEqual(output.dtype, np.float32)

        audio_stereo = np.array([[0.0, 1.0], [0.5, -0.5], [1.0, -1.0]], dtype=np.float32)
        output = format_audio(audio_stereo, sample_rate)
        expected = np.mean(audio_stereo, axis=1)
        np.testing.assert_array_almost_equal(output, expected, decimal=5)
        self.assertEqual(output.ndim, 1)

        from_sampling_rate = 32000
        duration_sec = 1.0
        n_samples = int(duration_sec * from_sampling_rate)
        audio_resample = np.linspace(-1.0, 1.0, num=n_samples, dtype=np.float32)
        output = format_audio(audio_resample, from_sampling_rate)
        expected_length = int(n_samples * 16000 / from_sampling_rate)
        self.assertEqual(output.shape[0], expected_length)
        self.assertEqual(output.dtype, np.float32)

        from_sampling_rate = 1000000
        audio_short = np.array([0.5], dtype=np.float32)
        output = format_audio(audio_short, from_sampling_rate)
        self.assertEqual(output.size, 0)
        self.assertEqual(output.dtype, np.int16)

        audio_mono_int8 = np.array([0, 64, -64, 127, -128], dtype=np.int8)
        output = format_audio(audio_mono_int8, sample_rate)
        expected = audio_mono_int8.astype(np.float32) / 128.0
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_wav_to_int16(self):
        from verbatim.audio.audio import wav_to_int16

        data_int16 = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        result = wav_to_int16(data_int16)
        np.testing.assert_array_equal(result, data_int16)

        data_float16 = np.array([0, 0.5, -0.5, 1, -1], dtype=np.float16)
        n = max(math.fabs(np.min(data_float16)), math.fabs(np.max(data_float16)))
        expected = (data_float16.astype(np.float32) / n * np.iinfo(np.int16).max).astype(np.int16)
        result = wav_to_int16(data_float16)
        np.testing.assert_array_equal(result, expected)

        with self.assertRaises(ValueError):
            wav_to_int16(np.array([0, 1, 2], dtype=np.uint8))

    def test_samples_to_seconds(self):
        from verbatim.audio.audio import samples_to_seconds
        self.assertEqual(samples_to_seconds(0), 0.0)
        self.assertEqual(samples_to_seconds(16000), 1.0)
        self.assertEqual(samples_to_seconds(8000), 0.5)
        self.assertTrue(math.isclose(samples_to_seconds(48000), 3.0, rel_tol=1e-6))

    def test_seconds_to_samples(self):
        from verbatim.audio.audio import seconds_to_samples
        self.assertEqual(seconds_to_samples(0), 0)
        self.assertEqual(seconds_to_samples(1), 16000)
        self.assertEqual(seconds_to_samples(0.5), 8000)
        self.assertEqual(seconds_to_samples(2.5), 40000)

    def test_seconds_to_timestr(self):
        from verbatim.audio.audio import seconds_to_timestr
        self.assertEqual(seconds_to_timestr(0.0), "[00:00:00.000]")
        self.assertEqual(seconds_to_timestr(1.5), "[00:00:01.500]")
        self.assertEqual(seconds_to_timestr(3661.78), "[01:01:01.780]")
        self.assertEqual(seconds_to_timestr(59.999), "[00:00:59.999]")

    def test_sample_to_timestr(self):
        from verbatim.audio.audio import sample_to_timestr
        self.assertEqual(sample_to_timestr(32000, 16000), "[00:00:02.000]")
        self.assertEqual(sample_to_timestr(8000, 16000), "[00:00:00.500]")
        self.assertEqual(sample_to_timestr(44100, 44100), "[00:00:01.000]")

    def test_timestr_to_samples(self):
        from verbatim.audio.audio import timestr_to_samples
        sample_rate = 16000
        self.assertEqual(timestr_to_samples("01:01:01.780", sample_rate), int((1 * 3600 + 1 * 60 + 1 + 0.780) * sample_rate))
        self.assertEqual(timestr_to_samples("01:01.500", sample_rate), int((1 * 60 + 1 + 0.500) * sample_rate))
        self.assertEqual(timestr_to_samples("1.500", sample_rate), int(1.5 * sample_rate))
        self.assertEqual(timestr_to_samples("01:05", sample_rate), int((1 * 60 + 5) * sample_rate))
        self.assertEqual(timestr_to_samples("5", sample_rate), int(5 * sample_rate))
        self.assertEqual(timestr_to_samples("  0:05.000  ", sample_rate), int(5 * sample_rate))
        with self.assertRaises(ValueError):
            timestr_to_samples("invalid_format")

if __name__ == "__main__":
    unittest.main()

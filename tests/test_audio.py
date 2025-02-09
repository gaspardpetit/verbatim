import math
import pytest
import numpy as np

# pylint: disable=import-outside-toplevel

def test_format_audio():
    from verbatim.audio.audio import format_audio

    # Case 1: Mono float32 input with correct sample rate (16 kHz) should be unchanged.
    sample_rate = 16000
    audio_mono_float = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    output = format_audio(audio_mono_float, sample_rate)
    np.testing.assert_array_almost_equal(output, audio_mono_float)
    assert output.dtype == np.float32

    # Case 2: Mono int16 input with 16 kHz sample rate should be normalized correctly.
    audio_mono_int16 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    output = format_audio(audio_mono_int16, sample_rate)
    expected = audio_mono_int16.astype(np.float32) / 32768.0
    np.testing.assert_array_almost_equal(output, expected, decimal=5)
    assert output.dtype == np.float32

    # Case 3: Stereo float32 input should be mixed down to mono.
    # Here, we assume the stereo shape is (num_samples, num_channels).
    audio_stereo = np.array([[0.0, 1.0],
                             [0.5, -0.5],
                             [1.0, -1.0]], dtype=np.float32)
    output = format_audio(audio_stereo, sample_rate)
    expected = np.mean(audio_stereo, axis=1)
    np.testing.assert_array_almost_equal(output, expected, decimal=5)
    assert output.ndim == 1

    # Case 4: Input with a sample rate different from 16 kHz should be resampled.
    # For instance, using 32 kHz input, the output length should be halved.
    from_sampling_rate = 32000
    duration_sec = 1.0  # 1 second of audio
    n_samples = int(duration_sec * from_sampling_rate)
    audio_resample = np.linspace(-1.0, 1.0, num=n_samples, dtype=np.float32)
    output = format_audio(audio_resample, from_sampling_rate)
    expected_length = int(n_samples * 16000 / from_sampling_rate)
    assert output.shape[0] == expected_length
    assert output.dtype == np.float32

    # Case 5: If resampling would yield zero samples, the function returns an empty array of dtype int16.
    # For example, a 1-sample input at an extremely high input sample rate.
    from_sampling_rate = 1000000  # Extremely high sample rate
    audio_short = np.array([0.5], dtype=np.float32)
    output = format_audio(audio_short, from_sampling_rate)
    assert output.size == 0
    assert output.dtype == np.int16

    # Optional Case 6: Mono int8 input should be normalized (divided by 128.0).
    audio_mono_int8 = np.array([0, 64, -64, 127, -128], dtype=np.int8)
    output = format_audio(audio_mono_int8, sample_rate)
    expected = audio_mono_int8.astype(np.float32) / 128.0
    np.testing.assert_array_almost_equal(output, expected, decimal=5)

def test_wav_to_int16():
    from verbatim.audio.audio import wav_to_int16

    # Case 1: np.int16 input should be returned as-is.
    data_int16 = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    result = wav_to_int16(data_int16)
    np.testing.assert_array_equal(result, data_int16)

    # Case 2: np.float16 input.
    data_float16 = np.array([0, 0.5, -0.5, 1, -1], dtype=np.float16)
    n = max(math.fabs(np.min(data_float16)), math.fabs(np.max(data_float16)))
    # Normalize and scale to int16 range.
    expected = (data_float16.astype(np.float32) / n * np.iinfo(np.int16).max).astype(np.int16)
    result = wav_to_int16(data_float16)
    np.testing.assert_array_equal(result, expected)

    # Case 3: np.float32 input.
    data_float32 = np.array([0, 0.5, -0.5, 1, -1], dtype=np.float32)
    n = max(math.fabs(np.min(data_float32)), math.fabs(np.max(data_float32)))
    expected = (data_float32 / n * np.iinfo(np.int16).max).astype(np.int16)
    result = wav_to_int16(data_float32)
    np.testing.assert_array_equal(result, expected)

    # Case 4: np.float64 input.
    data_float64 = np.array([0, 0.5, -0.5, 1, -1], dtype=np.float64)
    n = max(math.fabs(np.min(data_float64)), math.fabs(np.max(data_float64)))
    expected = (data_float64 / n * np.iinfo(np.int16).max).astype(np.int16)
    result = wav_to_int16(data_float64)
    np.testing.assert_array_equal(result, expected)

    # Case 5: np.int8 input.
    data_int8 = np.array([0, 64, -64, 127, -128], dtype=np.int8)
    scale_factor = (1.0 * np.iinfo(np.int16).max) / np.iinfo(np.int8).max
    expected = (data_int8 * scale_factor).astype(np.int16)
    result = wav_to_int16(data_int8)
    np.testing.assert_array_equal(result, expected)

    # Case 6: np.int32 input.
    data_int32 = np.array([0, 100000, -100000, np.iinfo(np.int32).max, np.iinfo(np.int32).min], dtype=np.int32)
    scale_factor = (1.0 * np.iinfo(np.int16).max) / np.iinfo(np.int32).max
    expected = (data_int32 * scale_factor).astype(np.int16)
    result = wav_to_int16(data_int32)
    np.testing.assert_array_equal(result, expected)

    # Case 7: Unsupported dtype (e.g., np.uint8) should raise a ValueError.
    with pytest.raises(ValueError):
        wav_to_int16(np.array([0, 1, 2], dtype=np.uint8))

def test_samples_to_seconds():
    """
    Test that the conversion from sample index to seconds works correctly
    using a fixed sampling rate of 16 kHz.
    """
    from verbatim.audio.audio import samples_to_seconds

    # 0 samples should yield 0 seconds.
    assert samples_to_seconds(0) == 0.0
    # 16,000 samples equals 1 second.
    assert samples_to_seconds(16000) == 1.0
    # 8,000 samples equals 0.5 seconds.
    assert samples_to_seconds(8000) == 0.5
    # 48,000 samples equals 3 seconds.
    assert math.isclose(samples_to_seconds(48000), 3.0, rel_tol=1e-6)

def test_seconds_to_samples():
    """
    Test that the conversion from seconds to sample index works correctly.
    """
    from verbatim.audio.audio import seconds_to_samples

    # 0 seconds should yield 0 samples.
    assert seconds_to_samples(0) == 0
    # 1 second corresponds to 16,000 samples.
    assert seconds_to_samples(1) == 16000
    # 0.5 seconds corresponds to 8,000 samples.
    assert seconds_to_samples(0.5) == 8000
    # 2.5 seconds corresponds to 40,000 samples.
    assert seconds_to_samples(2.5) == 40000

def test_seconds_to_timestr():
    """
    Test that a float representing seconds is correctly formatted to a
    time string of the form "[hh:mm:ss.mmm]".
    """
    from verbatim.audio.audio import seconds_to_timestr

    # 0 seconds should be "[00:00:00.000]".
    assert seconds_to_timestr(0.0) == "[00:00:00.000]"
    # 1.5 seconds should be "[00:00:01.500]".
    assert seconds_to_timestr(1.5) == "[00:00:01.500]"
    # 3661.78 seconds is 1 hour, 1 minute, 1 second, and 780 milliseconds.
    assert seconds_to_timestr(3661.78) == "[01:01:01.780]"
    # Edge-case: When the fractional part rounds up.
    # For 1.9995 seconds, math.floor(1.9995) yields 1, and
    # the milliseconds calculation is: round((0.9995 * 1000)) = 1000.
    # This will result in the string "[00:00:01.1000]".
    assert seconds_to_timestr(1.9995) == "[00:00:01.1000]"
    # Test another edge: 59.999 seconds.
    assert seconds_to_timestr(59.999) == "[00:00:59.999]"

def test_sample_to_timestr():
    """
    Test that a sample index is correctly converted to a time string given a
    specific sample rate.
    """
    from verbatim.audio.audio import sample_to_timestr, seconds_to_timestr, seconds_to_samples

    # For a 16 kHz sample rate:
    # 32,000 samples should equal 2 seconds.
    assert sample_to_timestr(32000, 16000) == "[00:00:02.000]"
    # 8,000 samples equals 0.5 seconds.
    assert sample_to_timestr(8000, 16000) == "[00:00:00.500]"

    # For a different sample rate, e.g., 44,100 Hz:
    # 44,100 samples should equal exactly 1 second.
    assert sample_to_timestr(44100, 44100) == "[00:00:01.000]"

    # Consistency check:
    # Convert a known seconds value to samples (at 16 kHz) and then back to a time string.
    seconds_value = 3661.78
    sample_index = seconds_to_samples(seconds_value)
    expected_timestr = seconds_to_timestr(seconds_value)
    assert sample_to_timestr(sample_index, 16000) == expected_timestr

def test_timestr_to_samples():
    from verbatim.audio.audio import timestr_to_samples
    sample_rate = 16000

    # Test hh:mm:ss.ms format.
    # "01:01:01.780" -> 1 hour, 1 minute, 1 second, and 780 milliseconds.
    total_seconds = 1 * 3600 + 1 * 60 + 1 + 0.780
    expected_samples = int(total_seconds * sample_rate)
    assert timestr_to_samples("01:01:01.780", sample_rate) == expected_samples

    # Test mm:ss.ms format.
    # "01:01.500" -> 1 minute, 1 second, and 500 milliseconds.
    total_seconds = 1 * 60 + 1 + 0.500
    expected_samples = int(total_seconds * sample_rate)
    assert timestr_to_samples("01:01.500", sample_rate) == expected_samples

    # Test ss.ms format with three-digit milliseconds.
    # "1.500" should be interpreted as 1 second and 500 milliseconds.
    total_seconds = 1 + 0.500
    expected_samples = int(total_seconds * sample_rate)
    assert timestr_to_samples("1.500", sample_rate) == expected_samples

    # Test ss.ms format with a single digit after the dot.
    # "1.5" is interpreted as 1 second and 5 milliseconds (i.e. 1.005 seconds), not 1.5 seconds.
    total_seconds = 1 + 0.005
    expected_samples = int(total_seconds * sample_rate)
    assert timestr_to_samples("1.005", sample_rate) == expected_samples

    # Test mm:ss format without milliseconds.
    # "01:05" -> 1 minute and 5 seconds.
    total_seconds = 1 * 60 + 5
    expected_samples = int(total_seconds * sample_rate)
    assert timestr_to_samples("01:05", sample_rate) == expected_samples

    # Test ss format (just seconds).
    # "5" -> 5 seconds.
    total_seconds = 5
    expected_samples = int(total_seconds * sample_rate)
    assert timestr_to_samples("5", sample_rate) == expected_samples

    # Test extra whitespace is correctly handled.
    total_seconds = 5
    expected_samples = int(total_seconds * sample_rate)
    assert timestr_to_samples("  0:05.000  ", sample_rate) == expected_samples

    # Test with a different sample rate (e.g., 44100 Hz) using "1.500" format.
    sample_rate_alt = 44100
    total_seconds = 1 + 0.500  # 1.5 seconds exactly.
    expected_samples = int(total_seconds * sample_rate_alt)
    assert timestr_to_samples("1.500", sample_rate_alt) == expected_samples

    expected_samples = int(1.5 * sample_rate)  # 24000 samples
    assert timestr_to_samples("1.5", sample_rate) == expected_samples

    # Test that an invalid format raises a ValueError.
    with pytest.raises(ValueError):
        timestr_to_samples("invalid_format")

import logging
import math
import re

import numpy as np
from numpy.typing import NDArray

from .settings import AUDIO_PARAMS

# Configure logger
LOG = logging.getLogger(__name__)


def format_audio(audio: NDArray, from_sampling_rate: int) -> NDArray:
    to_sampling_rate = AUDIO_PARAMS.sample_rate

    float_audio: NDArray
    if audio.dtype == np.float32:
        float_audio = audio
    else:
        if audio.dtype == np.int8:
            float_audio = audio.astype(np.float32) / 128.0
        elif audio.dtype == np.int16:
            float_audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            float_audio = audio.astype(np.float32) / 2147483648.0
        else:
            float_audio = audio.astype(np.float32)

    # If the audio is stereo, mix it down to mono
    mono_audio: NDArray
    if float_audio.ndim == 1:
        mono_audio = float_audio
    else:
        LOG.debug("Mixing %s channels down to mono.", float_audio.ndim)
        mono_audio = np.mean(float_audio, axis=1)

    # Resample if the audio sample rate is not 16 kHz
    resampled_audio: NDArray
    if from_sampling_rate == to_sampling_rate:
        resampled_audio = mono_audio
    else:
        LOG.debug("Resampling from %s Hz to %s Hz.", from_sampling_rate, to_sampling_rate)
        # Lazy import to avoid pulling scipy.signal during CLI startup
        from scipy.signal import resample  # type: ignore  # pylint: disable=import-outside-toplevel

        num_samples = int(len(mono_audio) * to_sampling_rate / from_sampling_rate)
        if num_samples == 0:
            return np.array([], dtype=np.int16)
        resampled_audio = resample(mono_audio, num_samples)  # pyright: ignore[reportAssignmentType]

    return resampled_audio.astype(np.float32)


def wav_to_int16(data):
    if data.dtype == np.int16:
        return data
    if data.dtype in (np.float16, np.float32, np.float64):
        # Use float32 math to avoid float16 precision issues when scaling
        float_data = data.astype(np.float32)
        min_val = float_data.min()
        max_val = float_data.max()
        n = max(math.fabs(min_val), math.fabs(max_val))
        if n == 0:
            return np.zeros_like(float_data, dtype=np.int16)
        scaled = (float_data / n) * np.iinfo(np.int16).max
        return scaled.astype(np.int16)
    if data.dtype == np.int8:
        return (data * ((1.0 * np.iinfo(np.int16).max) / np.iinfo(np.int8).max)).astype(np.int16)
    if data.dtype == np.int32:
        return (data * ((1.0 * np.iinfo(np.int16).max) / np.iinfo(np.int32).max)).astype(np.int16)
    raise ValueError(f"unexpected: {data.dtype}")


def samples_to_seconds(index: int) -> float:
    return index / AUDIO_PARAMS.sample_rate


def seconds_to_samples(seconds: float) -> int:
    return int(seconds * AUDIO_PARAMS.sample_rate)


def sample_to_timestr(sample: int, sample_rate: int):
    seconds: float = sample / sample_rate
    return seconds_to_timestr(seconds=seconds)


def seconds_to_timestr(seconds: float) -> str:
    hour_part = math.floor(seconds // 3600)
    minute_part = math.floor((seconds % 3600) // 60)
    second_part = math.floor(seconds % 60)
    ms_part = round((seconds - math.floor(seconds)) * 1000)  # Extract only milliseconds

    return f"[{hour_part:02}:{minute_part:02}:{second_part:02}.{ms_part:03}]"


def timestr_to_samples(timestr: str, sample_rate: int = AUDIO_PARAMS.sample_rate) -> int:
    """
    Converts a time string in the format hh:mm:ss.ms, mm:ss.ms, or ss.ms
    (where the fractional part represents fractional seconds) to the corresponding sample index.

    Args:
        timestr (str): Time string in the format hh:mm:ss.ms, mm:ss.ms, or ss.ms.
        sample_rate (int): Sampling rate in Hz (default is 16000).

    Returns:
        int: The corresponding sample index.
    """
    # Define regex patterns for specific formats.
    # Here, we capture the fractional part as a sequence of digits.
    hh_mm_ss_ms_pattern = re.compile(r"^(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)(?:\.(?P<fraction>\d+))?$")
    mm_ss_ms_pattern = re.compile(r"^(?P<minutes>\d+):(?P<seconds>\d+)(?:\.(?P<fraction>\d+))?$")
    ss_ms_pattern = re.compile(r"^(?P<seconds>\d+)(?:\.(?P<fraction>\d+))?$")

    timestr = timestr.strip()

    if match := hh_mm_ss_ms_pattern.match(timestr):
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        fraction_str = match.group("fraction") if match.group("fraction") else "0"
    elif match := mm_ss_ms_pattern.match(timestr):
        hours = 0
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        fraction_str = match.group("fraction") if match.group("fraction") else "0"
    elif match := ss_ms_pattern.match(timestr):
        hours = 0
        minutes = 0
        seconds = int(match.group("seconds"))
        fraction_str = match.group("fraction") if match.group("fraction") else "0"
    else:
        raise ValueError(f"Invalid time string format: {timestr}")

    # Interpret the fractional part correctly.
    # If fraction_str has N digits, then it represents fraction_str / (10 ** N) seconds.
    fraction = int(fraction_str) / (10 ** len(fraction_str)) if fraction_str else 0.0

    total_seconds = hours * 3600 + minutes * 60 + seconds + fraction

    return int(total_seconds * sample_rate)

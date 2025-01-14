import logging
import math
import re

import numpy as np
from numpy.typing import NDArray
from pydub import AudioSegment
from scipy.signal import resample

# Configure logger
LOG = logging.getLogger(__name__)


def format_audio(audio: NDArray, from_sampling_rate: int) -> NDArray:
    to_sampling_rate = 16000

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
        LOG.info(f"Mixing {float_audio.ndim} channels down to mono.")
        mono_audio = np.mean(float_audio, axis=1)

    # Resample if the audio sample rate is not 16 kHz
    resampled_audio: NDArray
    if from_sampling_rate == to_sampling_rate:
        resampled_audio = mono_audio
    else:
        LOG.info(f"Resampling from {from_sampling_rate} Hz to {to_sampling_rate} Hz.")
        num_samples = int(len(mono_audio) * to_sampling_rate / from_sampling_rate)
        if num_samples == 0:
            return np.array([], dtype=np.int16)
        resampled_audio = resample(mono_audio, num_samples)  # pyright: ignore[reportAssignmentType]

    return resampled_audio.astype(np.float32)


def wav_to_int16(data):
    if data.dtype == np.int16:
        return data
    if data.dtype == np.float16:
        min_val = np.min(data)
        max_val = np.max(data)
        n = max(math.fabs(min_val), math.fabs(max_val))
        data = data / n
        return (data * np.iinfo(np.int16).max).astype(np.int16)
    if data.dtype == np.float32:
        min_val = np.min(data)
        max_val = np.max(data)
        n = max(math.fabs(min_val), math.fabs(max_val))
        data = data / n
        return (data * np.iinfo(np.int16).max).astype(np.int16)
    if data.dtype == np.float64:
        min_val = np.min(data)
        max_val = np.max(data)
        n = max(math.fabs(min_val), math.fabs(max_val))
        data = data / n
        return (data * np.iinfo(np.int16).max).astype(np.int16)
    if data.dtype == np.int8:
        return (data * ((1.0 * np.iinfo(np.int16).max) / np.iinfo(np.int8).max)).astype(np.int16)
    if data.dtype == np.int32:
        return (data * ((1.0 * np.iinfo(np.int16).max) / np.iinfo(np.int32).max)).astype(np.int16)
    raise ValueError(f"unexpected: {data.dtype}")


def samples_to_seconds(index: int) -> float:
    return index / 16000


def seconds_to_samples(seconds: float) -> int:
    return int(seconds * 16000)


def timestr_to_samples(timestr: str, sample_rate: int = 16000) -> int:
    """
    Converts a time string in the format hh:mm:ss.ms, mm:ss.ms, or ss.ms
    (milliseconds optional) to the corresponding sample index.

    Args:
        timestr (str): Time string in the format hh:mm:ss.ms, mm:ss.ms, or ss.ms.
        sample_rate (int): Sampling rate in Hz (default is 16000).

    Returns:
        int: The corresponding sample index.
    """
    # Define regex patterns for specific formats
    hh_mm_ss_ms_pattern = re.compile(r"^(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$")
    mm_ss_ms_pattern = re.compile(r"^(?P<minutes>\d+):(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$")
    ss_ms_pattern = re.compile(r"^(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$")

    # Match the input against patterns
    if match := hh_mm_ss_ms_pattern.match(timestr.strip()):
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    elif match := mm_ss_ms_pattern.match(timestr.strip()):
        hours = 0
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    elif match := ss_ms_pattern.match(timestr.strip()):
        hours = 0
        minutes = 0
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    else:
        raise ValueError(f"Invalid time string format: {timestr}")

    # Calculate total time in seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    # Convert to sample index
    return int(total_seconds * sample_rate)


def convert_mp3_to_wav(input_mp3, output_wav):
    # Load the mp3 file
    audio = AudioSegment.from_mp3(input_mp3)
    # Export the audio as wav
    audio.export(output_wav, format="wav")

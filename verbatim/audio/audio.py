import logging
import math

import numpy as np
from pydub import AudioSegment
from scipy.signal import resample

# Configure logger
LOG = logging.getLogger(__name__)

def format_audio(audio: np.ndarray, from_sampling_rate: int) -> np.ndarray:
    to_sampling_rate = 16000

    if audio.dtype != np.float32:
        if audio.dtype == np.int8:
            audio = audio.astype(np.float32) / 128.0
        elif audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

    # If the audio is stereo, mix it down to mono
    if audio.ndim > 1:
        LOG.info(f"Mixing {audio.ndim} channels down to mono.")
        audio = np.mean(audio, axis=1)

    # Resample if the audio sample rate is not 16 kHz
    if from_sampling_rate != to_sampling_rate:
        LOG.info(f"Resampling from {from_sampling_rate} Hz to {to_sampling_rate} Hz.")
        num_samples = int(len(audio) * to_sampling_rate / from_sampling_rate)
        if num_samples == 0:
            return np.array([], dtype=np.int16)
        audio = resample(audio, num_samples)

    return audio.astype(np.float32)


def wav_to_int16(data):
    if data.dtype == np.int16:
        return data
    if data.dtype == np.float16:
        min_val = np.min(data)
        max_val = np.max(data)
        n = max(math.fabs(min_val),math.fabs(max_val))
        data = data / n
        return (data * np.iinfo(np.int16).max).astype(np.int16)
    if data.dtype == np.float32:
        min_val = np.min(data)
        max_val = np.max(data)
        n = max(math.fabs(min_val),math.fabs(max_val))
        data = data / n
        return (data * np.iinfo(np.int16).max).astype(np.int16)
    if data.dtype == np.float64:
        min_val = np.min(data)
        max_val = np.max(data)
        n = max(math.fabs(min_val),math.fabs(max_val))
        data = data / n
        return (data * np.iinfo(np.int16).max).astype(np.int16)
    if data.dtype == np.int8:
        return (data * ((1.0 * np.iinfo(np.int16).max) / np.iinfo(np.int8).max)).astype(np.int16)
    if data.dtype == np.int32:
        return (data * ((1.0 * np.iinfo(np.int16).max) / np.iinfo(np.int32).max)).astype(np.int16)
    raise ValueError(f"unexpected: {data.dtype}")

def samples_to_seconds(index:int):
    return index / 16000.0

def convert_mp3_to_wav(input_mp3, output_wav):
    # Load the mp3 file
    audio = AudioSegment.from_mp3(input_mp3)
    # Export the audio as wav
    audio.export(output_wav, format="wav")

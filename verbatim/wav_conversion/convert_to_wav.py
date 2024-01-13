from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
import soundfile as sf
import torchaudio
from pydub import AudioSegment


class ConvertToWav(ABC):
    @abstractmethod
    def execute(self, source_file_path: str, audio_file_path: str, **kwargs: dict) -> None:
        """
        Abstract method for converting audio files to WAV format.

        Args:
            source_file_path (str): Path to the input audio file.
            audio_file_path (str): Path to save the output WAV file.
            **kwargs (dict): Additional parameters for the conversion process.
        """

    @staticmethod
    def format_float32_16khz_mono_audio(audio: AudioSegment) -> ndarray:
        """
        Convert PyDub AudioSegment to a NumPy array with float32 format and 16kHz sample rate.

        Args:
            audio (AudioSegment): Input audio in PyDub's AudioSegment format.

        Returns:
            ndarray: NumPy array representing the audio with float32 format and 16kHz sample rate.
        """
        np_audio: ndarray = np.frombuffer(audio.set_channels(1).set_sample_width(2).raw_data, dtype=np.int16)
        waveform: Tensor = torch.from_numpy(np_audio).cuda()
        waveform = waveform.float() / 32767.0
        resampler = torchaudio.transforms.Resample(orig_freq=audio.frame_rate, new_freq=16000).cuda()
        waveform = resampler(waveform)
        return waveform.cpu().numpy()

    @staticmethod
    def load_float32_16khz_mono_audio(input_file: str) -> ndarray:
        """
        Load audio from file and convert it to a NumPy array with float32 format and 16kHz sample rate.

        Args:
            input_file (str): Path to the input audio file.

        Returns:
            ndarray: NumPy array representing the audio with float32 format and 16kHz sample rate.
        """
        audio: AudioSegment = AudioSegment.from_file(input_file).set_channels(1).set_sample_width(2)
        return ConvertToWav.format_float32_16khz_mono_audio(audio)

    @staticmethod
    def save_float32_16khz_mono_audio(waveform: ndarray, wav_file: str) -> None:
        """
        Save a NumPy array with float32 format and 16kHz sample rate to a WAV file.

        Args:
            waveform (ndarray): NumPy array representing the audio with float32 format and 16kHz sample rate.
            wav_file (str): Path to save the output WAV file.
        """
        scaled_data = waveform.astype(np.float32)
        sf.write(wav_file, scaled_data, 16000)

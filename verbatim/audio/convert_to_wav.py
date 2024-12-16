import numpy as np
import soundfile as sf
import torch
import torchaudio
from numpy import ndarray
from pydub import AudioSegment
from torch import Tensor


class ConvertToWav:
    @staticmethod
    def format_float32_16khz_mono_audio(audio: AudioSegment, device: str = "cuda") -> ndarray:
        """
        Convert PyDub AudioSegment to a NumPy array with float32 format and 16kHz sample rate.

        Args:
            device:
            audio (AudioSegment): Input audio in PyDub's AudioSegment format.

        Returns:
            ndarray: NumPy array representing the audio with float32 format and 16kHz sample rate.
        """
        if device == "cpu":
            np_audio: ndarray = np.frombuffer(audio.set_channels(1).set_sample_width(2).raw_data, dtype=np.int16)
            waveform: Tensor = torch.from_numpy(np_audio)
            waveform = waveform.float() / 32767.0
            resampler = torchaudio.transforms.Resample(orig_freq=audio.frame_rate, new_freq=16000)
            waveform = resampler(waveform)
            return waveform.numpy()
        else:
            np_audio: ndarray = np.frombuffer(audio.set_channels(1).set_sample_width(2).raw_data, dtype=np.int16)
            waveform: Tensor = torch.from_numpy(np_audio).cuda()
            waveform = waveform.float() / 32767.0
            resampler = torchaudio.transforms.Resample(orig_freq=audio.frame_rate, new_freq=16000).cuda()
            waveform = resampler(waveform)
            return waveform.cpu().numpy()

    @staticmethod
    def load_float32_16khz_mono_audio(input_file: str, device:str = "cuda") -> ndarray:
        """
        Load audio from file and convert it to a NumPy array with float32 format and 16kHz sample rate.

        Args:
            device:
            input_file (str): Path to the input audio file.

        Returns:
            ndarray: NumPy array representing the audio with float32 format and 16kHz sample rate.
        """
        audio: AudioSegment = AudioSegment.from_file(input_file).set_channels(1).set_sample_width(2)
        return ConvertToWav.format_float32_16khz_mono_audio(audio=audio, device=device)

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

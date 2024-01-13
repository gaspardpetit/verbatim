from numpy import ndarray

from .isolate_voices import IsolateVoices
from ..wav_conversion import ConvertToWav


class IsolateVoicesFile(IsolateVoices):
    """
    Voice isolation using an existing audio file.

    This class inherits from IsolateVoices and implements the voice isolation process using an existing audio file.
    It loads the audio file, returning its waveform as a NumPy array.

    Attributes:
        None
    """

    def execute(self, audio_file_path: str, voice_file_path: str, **kwargs) -> ndarray:
        """
        Execute the voice isolation process using an existing audio file.

        Args:
            audio_file_path (str): Path to the existing audio file for voice isolation (not used in this method)..
            voice_file_path (str): Path to save the isolated voice audio file.
            **kwargs: Additional parameters (not used in this method).

        Returns:
            ndarray: NumPy array representing the isolated voice audio.
        """
        # Load the existing audio file as a NumPy array
        waveform: ndarray = ConvertToWav.load_float32_16khz_mono_audio(voice_file_path, kwargs['device'])

        return waveform

from numpy import ndarray

from ..wav_conversion import ConvertToWav
from .isolate_voices import IsolateVoices


class IsolateVoicesNone(IsolateVoices):
    """
    Passthrough voice isolation.

    This class inherits from IsolateVoices and represents a passthrough implementation where the input audio
    is loaded and saved without any voice isolation.

    Attributes:
        None
    """

    def execute(self, audio_file_path: str, voice_file_path: str, **kwargs) -> ndarray:
        """
        Execute the passthrough voice isolation.

        Args:
            audio_file_path (str): Path to the input audio file.
            voice_file_path (str): Path to save the output audio file.
            **kwargs: Additional parameters (not used in this method).

        Returns:
            ndarray: NumPy array representing the input audio.
        """
        # Load the input audio as a NumPy array
        waveform: ndarray = ConvertToWav.load_float32_16khz_mono_audio(audio_file_path, kwargs['device'])

        # Save the input audio to the destination path
        ConvertToWav.save_float32_16khz_mono_audio(waveform, voice_file_path)

        return waveform

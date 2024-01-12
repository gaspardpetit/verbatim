from .isolate_voices import IsolateVoices
from numpy import ndarray
from verbatim.wav_conversion import ConvertToWav


class IsolateVoicesNone(IsolateVoices):
    """
    Passthrough voice isolation.

    This class inherits from IsolateVoices and represents a passthrough implementation where the input audio
    is loaded and saved without any voice isolation.

    Attributes:
        None
    """

    def execute(self, source_path: str, destination_path: str, **kwargs) -> ndarray:
        """
        Execute the passthrough voice isolation.

        Args:
            source_path (str): Path to the input audio file.
            destination_path (str): Path to save the output audio file.
            **kwargs: Additional parameters (not used in this method).

        Returns:
            ndarray: NumPy array representing the input audio.
        """
        # Load the input audio as a NumPy array
        waveform: ndarray = ConvertToWav.load_float32_16khz_mono_audio(source_path)

        # Save the input audio to the destination path
        ConvertToWav.save_float32_16khz_mono_audio(waveform, destination_path)

        return waveform

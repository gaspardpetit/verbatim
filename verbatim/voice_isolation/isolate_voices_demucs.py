import os
from numpy import ndarray
import demucs.separate

from ..wav_conversion import ConvertToWav
from .isolate_voices import IsolateVoices


class IsolateVoicesDemucs(IsolateVoices):
    """
    Voice isolation using the Demucs algorithm.

    This class inherits from IsolateVoices and implements the voice isolation process using the Demucs algorithm.

    Attributes:
        None
    """

    def execute(self, audio_file_path: str, voice_file_path: str, **kwargs) -> ndarray:
        """
        Execute the voice isolation process using Demucs.

        Args:
            audio_file_path (str): Path to the source audio file.
            voice_file_path (str): Path to save the isolated voice audio file.
            **kwargs: Additional parameters (not used in this method).

        Returns:
            ndarray: NumPy array representing the isolated voice audio.
        """
        # Use Demucs to separate vocals from the source audio
        demucs.separate.main([audio_file_path, "--two-stems", "vocals", "--name", "htdemucs_ft"])

        # Build the path to the separated vocals file
        filename_noext = os.path.splitext(os.path.basename(audio_file_path))[0]
        source_path_separated = f"separated/htdemucs_ft/{filename_noext}/vocals.wav"

        # Load the separated vocals as a NumPy array
        waveform: ndarray = ConvertToWav.load_float32_16khz_mono_audio(source_path_separated)

        # Save the isolated vocals to the destination path
        ConvertToWav.save_float32_16khz_mono_audio(waveform, voice_file_path)

        return waveform

import shutil
import logging
from numpy import ndarray
from audio_separator.separator import Separator

from ..wav_conversion import ConvertToWav
from .isolate_voices import IsolateVoices

class IsolateVoicesMDX(IsolateVoices):
    """
    Voice isolation using the MDX algorithm.

    This class inherits from IsolateVoices and implements the voice isolation process using the MDX algorithm.

    Attributes:
        None
    """

    def execute(self, audio_file_path: str, voice_file_path: str, **kwargs) -> ndarray:
        """
        Execute the voice isolation process using the MDX algorithm.

        Args:
            audio_file_path (str): Path to the source audio file.
            voice_file_path (str): Path to save the isolated voice audio file.
            **kwargs: Additional parameters (not used in this method).

        Returns:
            ndarray: NumPy array representing the isolated voice audio.
        """
        # Initialize the MDX separator
        separator = Separator(log_level=kwargs['log_level'] or logging.WARNING)
        separator.load_model('Kim_Vocal_2')

        # Use MDX to separate vocals from the source audio
        output_file_paths = separator.separate(audio_file_path)

        # Move the generated files to the output directory
        instrument_audio = output_file_paths[0]
        voice_audio = output_file_paths[1]
        shutil.move(instrument_audio, f"{voice_file_path}-noise.wav")
        shutil.move(voice_audio, f"{voice_file_path}-voice.wav")

        # Load the separated vocals as a NumPy array
        waveform: ndarray = ConvertToWav.load_float32_16khz_mono_audio(
            f"{voice_file_path}-voice.wav", kwargs['device'])

        # Save the isolated vocals to the destination path
        ConvertToWav.save_float32_16khz_mono_audio(waveform, voice_file_path)

        return waveform

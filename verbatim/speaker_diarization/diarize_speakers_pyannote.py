from .diarize_speakers import DiarizeSpeakers
from pyannote.core import Annotation
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Pipeline
import torchaudio
import torch
import logging
import os

LOG = logging.getLogger(__name__)


class DiarizeSpeakersPyannote(DiarizeSpeakers):
    """
    Diarization implementation using PyAnnote for speaker segmentation.

    This class inherits from DiarizeSpeakers and implements diarization using PyAnnote for speaker segmentation.

    Attributes:
        None
    """

    def diarize_on_silences(self, audio_file: str, huggingface_token: str) -> Annotation:
        """
        Diarize speakers based on silences using PyAnnote.

        Args:
            audio_file (str): Path to the input audio file.
            huggingface_token (str): Hugging Face token for model access.

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # ... (unchanged)

    @staticmethod
    def _load_audio(audio_path: str) -> tuple:
        """
        Loads the audio waveform and sample rate from an audio file.

        Args:
            audio_path (str): Path to the input audio file.

        Returns:
            tuple: Tuple containing waveform and sample rate.
        """
        # ... (unchanged)

    @staticmethod
    def _get_device() -> torch.device:
        """
        Retrieves the device (CPU or CUDA) to use for diarization.

        Returns:
            torch.device: Device to use for diarization.
        """
        # ... (unchanged)

    @staticmethod
    def _load_pipeline(device: torch.device, huggingface_token: str) -> Pipeline:
        """
        Loads the diarization pipeline.

        Args:
            device (torch.device): Device to use for diarization.
            huggingface_token (str): Hugging Face token for model access.

        Returns:
            Pipeline: PyAnnote diarization pipeline.
        """
        # ... (unchanged)

    def diarize_on_speakers(self, audio_file: str, num_speakers: int, huggingface_token: str) -> Annotation:
        """
        Diarize speakers based on speaker count using PyAnnote.

        Args:
            audio_file (str): Path to the input audio file.
            num_speakers (int): Number of speakers to consider.
            huggingface_token (str): Hugging Face token for model access.

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # ... (unchanged)

    def execute(self, audio_file: str, rttm_file: str, min_speakers: int = 1, max_speakers: int = None,
                TOKEN_HUGGINGFACE: str = None, **kwargs: dict) -> Annotation:
        """
        Execute the diarization process using PyAnnote.

        Args:
            audio_file (str): Path to the input audio file.
            rttm_file (str): Path to the output RTTM (Rich Transcription Time Marked) file.
            min_speakers (int, optional): Minimum number of expected speakers. Default is 1.
            max_speakers (int, optional): Maximum number of expected speakers. Default is None (unbounded).
            TOKEN_HUGGINGFACE (str, optional): Hugging Face token for customization. Default is None.
            **kwargs (dict): Additional parameters (not used in this method).

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # ... (unchanged)

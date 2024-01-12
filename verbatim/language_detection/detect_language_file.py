from .detect_language import DetectLanguage
from ..transcription import Transcription
import logging
from numpy import ndarray
from pyannote.core import Annotation

LOG = logging.getLogger(__name__)


class DetectLanguageFile(DetectLanguage):
    """
    Class for loading language detection results from a file.

    Attributes:
        None
    """

    def execute_segment(self, speaker: str, speech_offset: float, speech_segment_float32_16khz: ndarray,
                        languages=None, **kwargs: dict) -> Transcription:
        """
        Placeholder method for executing language detection on a speech segment.

        Args:
            speaker (str): The speaker identifier.
            speech_offset (float): The offset in seconds from the beginning of the audio.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            languages (Optional[List[str]]): List of target languages for detection.
            **kwargs (dict): Additional keyword arguments for customization.

        Raises:
            NotImplementedError: This method is not implemented for file-based language detection.

        Returns:
            Transcription: Placeholder return value.
        """
        raise NotImplementedError

    def execute(self, diarization: Annotation, speech_segment_float32_16khz: ndarray,
                language_file: str, languages=None, **kwargs: dict) -> Transcription:
        """
        Loads language detection results from a file.

        Args:
            diarization (Annotation): Speaker diarization information (unused in this case).
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz (unused in this case).
            language_file (str): File path to load the detected language information.
            languages (Optional[List[str]]): List of target languages (unused in this case).
            **kwargs (dict): Additional keyword arguments (unused in this case).

        Returns:
            Transcription: Transcription object containing the loaded language information.
        """
        transcription = Transcription.load(language_file)
        return transcription

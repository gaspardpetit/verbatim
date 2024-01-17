import logging
from numpy import ndarray

from .transcribe_speech import TranscribeSpeech
from ..transcription import Transcription

LOG = logging.getLogger(__name__)

class TranscribeSpeechFile(TranscribeSpeech):
    """
    Implementation of TranscribeSpeech that loads transcriptions from a file.

    Attributes:
        None
    """

    def execute_segment(self, speech_segment_float32_16khz: ndarray,
                        speaker: str = "speaker", speech_offset: float = 0,
                        language: str = None, prompt: str = "",
                        **kwargs: dict) -> Transcription:
        """
        Execute transcription on a speech segment (Not implemented).

        Args:
            speaker (str): The speaker identifier.
            speech_offset (float): The offset in seconds from the beginning of the audio.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            language (str): Target language for transcription.
            prompt (str): Optional transcription prompt.
            **kwargs (dict): Additional keyword arguments for customization.

        Raises:
            NotImplementedError: This method is not implemented for loading from a file.

        Returns:
            None
        """
        raise NotImplementedError("execute_segment method is not implemented for loading from a file.")

    # pylint: disable=arguments-differ
    def execute(self, transcription_path: str, **kwargs: dict) -> Transcription:
        """
        Load transcriptions from a file.

        Args:
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            detected_languages (Transcription): Transcription containing detected languages.
            transcription_path (str): File path for loading transcriptions.
            diarization (Annotation): Diarization information.
            languages (list): List of target languages.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription loaded from the specified file.
        """
        transcription = Transcription.load(transcription_path)
        return transcription

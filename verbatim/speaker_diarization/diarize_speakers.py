from abc import ABC, abstractmethod
from pyannote.core import Annotation


class DiarizeSpeakers(ABC):
    """
    Abstract class for diarization of speakers in an audio file.

    This class defines the interface for diarization methods. Subclasses must implement the 'execute' method.

    Attributes:
        None
    """

    @abstractmethod
    def execute(self, audio_file: str, rttm_file: str, min_speakers: int = 1, max_speakers: int = None,
                **kwargs: dict) -> Annotation:
        """
        Execute the diarization process.

        Args:
            audio_file (str): Path to the input audio file.
            rttm_file (str): Path to the output RTTM (Rich Transcription Time Marked) file.
            min_speakers (int, optional): Minimum number of expected speakers. Default is 1.
            max_speakers (int, optional): Maximum number of expected speakers. Default is None (unbounded).
            **kwargs (dict): Additional parameters for customization.

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """


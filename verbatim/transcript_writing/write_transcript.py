from abc import ABC, abstractmethod
from ..transcription import Transcription


class WriteTranscript(ABC):
    """
    Abstract base class for writing transcriptions to a file.

    Attributes:
        None

    Methods:
        execute(transcript: Transcription, output_file: str, **kwargs: dict) -> None:
            Abstract method for executing the transcription writing process.

    """

    @abstractmethod
    def execute(self, transcript: Transcription, output_file: str, **kwargs: dict) -> None:
        """
        Execute the transcription writing process.

        Args:
            transcript (Transcription): The transcription to be written.
            output_file (str): The path to the output file.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            None

        Raises:
            NotImplementedError: This method must be implemented by the derived class.
        """

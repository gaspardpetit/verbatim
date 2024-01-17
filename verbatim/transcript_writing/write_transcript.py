from abc import abstractmethod
from ..filter import Filter

class WriteTranscript(Filter):
    """
    Abstract base class for writing transcriptions to a file.

    Attributes:
        None

    Methods:
        execute(transcript: Transcription, output_file: str, **kwargs: dict) -> None:
            Abstract method for executing the transcription writing process.

    """

    @abstractmethod
    # pylint: disable=arguments-differ
    def execute(self, transcription_path: str, output_file: str, **kwargs: dict) -> None:
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

from abc import ABC, abstractmethod
from numpy import ndarray


class IsolateVoices(ABC):
    """
    Abstract base class for voice isolation methods.

    This class defines the structure for voice isolation methods. Subclasses must implement the 'execute' method.

    Attributes:
        None
    """

    @abstractmethod
    def execute(self, audio_file_path: str, voice_file_path: str, **kwargs) -> ndarray:
        """
        Execute the voice isolation process.

        Args:
            audio_file_path (str): Path to the source audio file.
            voice_file_path (str): Path to save the isolated voice audio file.
            **kwargs: Additional parameters for customization.

        Returns:
            ndarray: NumPy array representing the isolated voice audio.
        """

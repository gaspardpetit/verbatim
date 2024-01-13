from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from .diarize_speakers import DiarizeSpeakers


class DiarizeSpeakersFile(DiarizeSpeakers):
    """
    Diarization implementation that loads speaker information from an existing RTTM file.

    This class inherits from DiarizeSpeakers and implements diarization by loading speaker 
    information from a pre-existing RTTM (Rich Transcription Time Marked) file.

    Attributes:
        None
    """

    def execute(self, voice_file_path: str, diarization_file: str, min_speakers: int = 1, max_speakers: int = None,
                **kwargs: dict) -> Annotation:
        """
        Execute the diarization process by loading speaker information from an RTTM file.

        Args:
            voice_file_path (str): Path to the input audio file (not used in this method).
            diarization_file (str): Path to the RTTM file containing speaker information.
            min_speakers (int, optional): Minimum number of expected speakers. Default is 1.
            max_speakers (int, optional): Maximum number of expected speakers. Default is None (unbounded).
            TOKEN_HUGGINGFACE (str, optional): Hugging Face token for customization. Default is None.
            **kwargs (dict): Additional parameters (not used in this method).

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # Load speaker information from the provided RTTM file
        rttms = load_rttm(diarization_file)

        # Return the first speaker information available
        return next(iter(rttms.values()))

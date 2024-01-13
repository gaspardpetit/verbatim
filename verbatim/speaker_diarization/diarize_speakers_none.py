import logging
from pyannote.core import Annotation, Segment
from pydub import AudioSegment

from .diarize_speakers import DiarizeSpeakers

LOG = logging.getLogger(__name__)


class DiarizeSpeakersNone(DiarizeSpeakers):
    """
    Diarization implementation that considers the entire audio as a single diarization segment.

    This class inherits from DiarizeSpeakers and implements diarization by considering the entire audio
    as a single diarization segment, assigned to a default speaker label.

    Attributes:
        None
    """

    def execute(self, voice_file_path: str, diarization_file: str, min_speakers: int = 1, max_speakers: int = None,
                **kwargs: dict) -> Annotation:
        """
        Execute the diarization process by considering the entire audio as a single diarization segment.

        Args:
            voice_file_path (str): Path to the input audio file.
            diarization_file (str): Path to the output RTTM (Rich Transcription Time Marked) file.
            min_speakers (int, optional): Minimum number of expected speakers. Default is 1.
            max_speakers (int, optional): Maximum number of expected speakers. Default is None (unbounded).
            **kwargs (dict): Additional parameters (not used in this method).

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # Initialize an empty diarization annotation
        diarization: Annotation = Annotation()

        # Load the audio as a Pydub AudioSegment
        sound: AudioSegment = AudioSegment.from_file(voice_file_path)

        # Create a diarization segment covering the entire audio with a default speaker label
        diarization[Segment(0, sound.duration_seconds)] = "Speaker"

        # Write the diarization to the provided RTTM file
        with open(diarization_file, "w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        # Log the diarization for information
        LOG.info(diarization)

        return diarization

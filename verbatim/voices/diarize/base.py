from abc import ABC, abstractmethod
import logging
from pyannote.core.annotation import Annotation

LOG = logging.getLogger(__name__)


class DiarizationStrategy(ABC):
    """Base class for all diarization strategies"""

    @abstractmethod
    def compute_diarization(self, file_path: str, out_rttm_file: str = None, **kwargs) -> Annotation:
        """
        Compute speaker diarization for the given audio file.

        Args:
            file_path: Path to the audio file
            out_rttm_file: Optional path to output RTTM file
            **kwargs: Strategy-specific parameters

        Returns:
            PyAnnote Annotation object containing speaker segments
        """
        pass

    def save_rttm(self, annotation: Annotation, out_rttm_file: str):
        """Save annotation to RTTM file"""
        with open(out_rttm_file, "w", encoding="utf-8") as rttm:
            annotation.write_rttm(rttm)

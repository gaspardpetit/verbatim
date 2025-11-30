import logging
import os
from typing import Optional

from verbatim_rttm import Annotation, load_rttm, loads_rttm

from .diarize.factory import create_diarizer

# Configure logger
LOG = logging.getLogger(__name__)

UNKNOWN_SPEAKER = "SPEAKER"


class Diarization:
    def __init__(self, device: str, huggingface_token: str):
        self.device = device
        self.huggingface_token = huggingface_token
        self.diarizer = None

    def __enter__(self) -> "Diarization":
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        del self.diarizer
        return False

    @staticmethod
    def load_diarization(rttm_file: str):
        if not os.path.exists(rttm_file):
            raise FileNotFoundError(f"RTTM file not found: {rttm_file}")

        annotation: Annotation = load_rttm(rttm_file)
        if len(annotation) == 0:
            raise ValueError(f"No diarization data found in RTTM file: {rttm_file}")
        return annotation

    @staticmethod
    def load_diarization_from_text(rttm_text: str) -> Annotation:
        """Load diarization from RTTM/VTTM content (text)."""
        annotation: Annotation = loads_rttm(rttm_text)
        if len(annotation) == 0:
            raise ValueError("No diarization data found in RTTM text")
        return annotation

    def compute_diarization(self, file_path: str, out_rttm_file: Optional[str] = None, strategy: str = "pyannote", **kwargs) -> Annotation:
        """
        Compute diarization using the specified strategy.

        # dump the diarization output to disk using RTTM format
        with open(out_rttm_file, "w", encoding="utf-8") as rttm:
            diarization.write_rttm(rttm)
        Args:
            file_path: Path to audio file
            out_rttm_file: Output RTTM file path
            strategy: Diarization strategy to use ('pyannote' or 'stereo')
            **kwargs: Strategy-specific parameters
        """
        self.diarizer = create_diarizer(strategy=strategy, device=self.device, huggingface_token=self.huggingface_token)

        return self.diarizer.compute_diarization(file_path=file_path, out_rttm_file=out_rttm_file, **kwargs)

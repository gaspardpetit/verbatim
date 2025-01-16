import logging
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod

from pyannote.core.annotation import Annotation

# Configure logger
LOG = logging.getLogger(__name__)

class SeparationStrategy(ABC):
    def __init__(self):
        pass

    def __enter__(self) -> "SeparationStrategy":
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        return False

    @abstractmethod
    def separate_speakers(
        self,
        *,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_speaker_wav_prefix="",
        nb_speakers: Optional[int] = None,
    ) -> Tuple[Annotation, Dict[str, str]]:
        pass

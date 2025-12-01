import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from verbatim_audio.sources.audiosource import AudioSource

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
        out_vttm_file: Optional[str] = None,
        out_speaker_wav_prefix="",
        nb_speakers: Optional[int] = None,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
    ) -> List[AudioSource]:
        pass

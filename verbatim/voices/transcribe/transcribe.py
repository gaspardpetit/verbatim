import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from numpy.typing import NDArray
from ...transcript.words import Word

LOG = logging.getLogger(__name__)

PREPEND_PUNCTUATIONS = "\"'“¿([{-"
APPEND_PUNCTUATIONS = "\"'.。,;，!！?？:：”)]}、"


class WhisperConfig:
    def __init__(self):
        self.prepend_punctuations = PREPEND_PUNCTUATIONS
        self.append_punctuations = APPEND_PUNCTUATIONS
        self.task = "transcribe"
        self.length_penalty = 1.0
        self.suppress_tokens = [-1]
        self.suppress_blank = False
        self.no_speech_threshold = 0.95
        self.logprob_threshold = -1.0
        self.compression_ratio_threshold = 2.4


class Transcriber(ABC):
    @abstractmethod
    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        pass

    @abstractmethod
    def transcribe(
        self,
        *,
        audio: NDArray,
        lang: str,
        prompt: str,
        prefix: str,
        window_ts: int,
        audio_ts: int,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Optional[List[float]] = None,
    ) -> List[Word]:
        pass

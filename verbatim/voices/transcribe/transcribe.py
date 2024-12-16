import logging
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from ...transcript.words import VerbatimWord

LOG = logging.getLogger(__name__)

PREPEND_PUNCTUATIONS="\"'“¿([{-"
APPEND_PUNCTUATIONS="\"'.。,;，!！?？:：”)]}、"


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

class Transcriber:
    @abstractmethod
    def guess_language(self, audio:np.array, lang:List[str]) -> Tuple[str, float]:
        pass

    @abstractmethod
    def transcribe(self, *, audio:np.array, lang: str, prompt: str, prefix: str, window_ts:int, audio_ts:int) -> List[VerbatimWord]:
        pass

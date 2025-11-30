from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from numpy.typing import NDArray

# Callbacks are intentionally simple so they can be implemented without heavy deps.
VadFn = Callable[[NDArray, int, int], List[Dict[str, int]]]
GuessLanguageFn = Callable[[NDArray, List[str]], Tuple[str, float]]


@dataclass
class LanguageDetectionRequest:
    audio: NDArray
    lang: List[str]
    timestamp: int
    window_ts: int
    audio_ts: int


@dataclass
class LanguageDetectionResult:
    language: str
    probability: float
    samples_used: int


@dataclass
class TranscriptionWindowResult:
    utterance: object
    unacknowledged: List[object]
    unconfirmed_words: List[object]

    def as_tuple(self):
        return self.utterance, self.unacknowledged, self.unconfirmed_words

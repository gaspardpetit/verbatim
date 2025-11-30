from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from numpy.typing import NDArray

# Minimal contract for transcription backends; lets us swap local/remote implementations.
if TYPE_CHECKING:
    from ..transcript.words import Word
else:  # pragma: no cover - type-only fallback to avoid heavy imports
    Word = object  # pylint: disable=invalid-name


class TranscriberProtocol(Protocol):
    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]: ...

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
    ) -> List[Word]: ...


class VadProtocol(Protocol):
    def __call__(self, audio: NDArray, min_speech_duration_ms: int, min_silence_duration_ms: int) -> List[Dict[str, int]]: ...


# Callbacks are intentionally simple so they can be implemented without heavy deps.
VadFn = VadProtocol
GuessLanguageFn = Callable[[NDArray, List[str]], Tuple[str, float]]

TUtt_co = TypeVar("TUtt_co", covariant=True)  # pylint: disable=invalid-name
TWord_co = TypeVar("TWord_co", covariant=True)  # pylint: disable=invalid-name


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
class TranscriptionWindowResult(Generic[TUtt_co, TWord_co]):
    utterance: TUtt_co
    unacknowledged: Sequence[TUtt_co]
    unconfirmed_words: Sequence[TWord_co]

    def as_tuple(self):
        return self.utterance, list(self.unacknowledged), list(self.unconfirmed_words)

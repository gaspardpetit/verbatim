from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol

from verbatim.transcript.words import Utterance, Word

StatusProgressUnit = Literal["percent", "count", "seconds"]


@dataclass(frozen=True)
class StatusProgress:
    current: float
    units: StatusProgressUnit
    start: float = 0.0
    finish: Optional[float] = None


@dataclass(frozen=True)
class StatusUpdate:
    state: Optional[str]
    progress: Optional[StatusProgress] = None
    utterance: Optional[Utterance] = None
    unacknowledged_utterances: Optional[List[Utterance]] = None
    unconfirmed_words: Optional[List[Word]] = None


class StatusHook(Protocol):
    def __call__(self, update: StatusUpdate) -> None: ...

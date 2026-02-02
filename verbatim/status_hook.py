import sys
from dataclasses import dataclass
from typing import BinaryIO, List, Literal, Optional, Protocol

from verbatim.transcript.words import Utterance, Word
from verbatim_files.format.stdout import StdoutTranscriptWriter
from verbatim_files.format.writer import TranscriptWriterConfig

StatusProgressUnit = Literal["percent", "count", "seconds"]


@dataclass(frozen=True)
class StatusProgress:
    current: float
    units: StatusProgressUnit
    start: float = 0.0
    finish: Optional[float] = None


@dataclass(frozen=True)
class StatusUpdate:
    state: str
    progress: Optional[StatusProgress] = None
    utterance: Optional[Utterance] = None
    unacknowledged_utterances: Optional[List[Utterance]] = None
    unconfirmed_words: Optional[List[Word]] = None


class StatusHook(Protocol):
    def __call__(self, update: StatusUpdate) -> None: ...


class SimpleProgressHook:
    def __init__(
        self,
        *,
        config: TranscriptWriterConfig,
        with_colours: bool = True,
        output: Optional[BinaryIO] = None,
    ):
        self._writer = StdoutTranscriptWriter(config=config, with_colours=with_colours)
        self._output = output or sys.stdout.buffer

    def __call__(self, update: StatusUpdate) -> None:
        if update.utterance is None:
            return
        payload = self._writer.format_utterance(
            utterance=update.utterance,
            unacknowledged_utterance=update.unacknowledged_utterances,
            unconfirmed_words=update.unconfirmed_words,
        )
        if not payload:
            return
        self._output.write(payload)
        if hasattr(self._output, "flush"):
            self._output.flush()

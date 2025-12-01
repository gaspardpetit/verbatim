from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from ..audio.audio import samples_to_seconds, seconds_to_samples

if TYPE_CHECKING:
    from faster_whisper.transcribe import Word as WhisperWord
    from pywhispercpp.model import Segment


@dataclass
class Word:
    start_ts: int
    end_ts: int
    word: str
    probability: float
    lang: str

    @classmethod
    def from_word(cls, word: "WhisperWord", lang: str, ts_offset: int = 0) -> "Word":
        """Creates a Word instance from a Word object with a timestamp offset."""
        # Lazy import to avoid pulling faster_whisper during CLI startup.

        start_ts = int(word.start * 16000) + ts_offset
        end_ts = int(word.end * 16000) + ts_offset
        return cls(
            start_ts=start_ts,
            lang=lang,
            end_ts=end_ts,
            word=word.word,
            probability=word.probability,
        )

    @classmethod
    def from_whisper_cpp_1w_segment(cls, segment: "Segment", lang: str, ts_offset: int = 0) -> "Word":
        """Creates a Word instance from a WhisperCPP 1-word segment with a timestamp offset."""
        # Lazy import to avoid pulling pywhispercpp during CLI startup.

        start_ts = seconds_to_samples(segment.t0 / 100) + ts_offset
        end_ts = seconds_to_samples(segment.t1 / 100) + ts_offset
        return cls(
            start_ts=start_ts,
            lang=lang,
            end_ts=end_ts,
            word=f" {segment.text}",
            probability=1.0,
        )


@dataclass
class Utterance:
    utterance_id: str
    speaker: Optional[str]
    start_ts: int
    end_ts: int
    text: str
    words: List[Word] = field(default_factory=list)

    def get_start(self) -> float:
        return samples_to_seconds(self.start_ts)

    def get_end(self) -> float:
        return samples_to_seconds(self.end_ts)

    @classmethod
    def from_words(cls, utterance_id: str, words: List[Word], speaker: Optional[str] = None) -> "Utterance":
        start_ts = words[0].start_ts
        end_ts = words[-1].end_ts
        text = "".join([w.word for w in words])
        return cls(utterance_id=utterance_id, start_ts=start_ts, end_ts=end_ts, words=words, text=text, speaker=speaker)

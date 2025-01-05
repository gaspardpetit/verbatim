from dataclasses import dataclass, field
from typing import List

from faster_whisper.transcribe import Word

from ..audio.audio import samples_to_seconds


@dataclass
class VerbatimWord:
    start_ts: int
    end_ts: int
    word: str
    probability: float
    lang: str

    @classmethod
    def from_word(cls, word: Word, lang: str, ts_offset: int = 0) -> "VerbatimWord":
        """Creates a VerbatimWord instance from a Word object with a timestamp offset."""
        start_ts = int(word.start * 16000) + ts_offset
        end_ts = int(word.end * 16000) + ts_offset
        return cls(
            start_ts=start_ts,
            lang=lang,
            end_ts=end_ts,
            word=word.word,
            probability=word.probability,
        )


@dataclass
class VerbatimUtterance:
    speaker: str
    start_ts: int
    end_ts: int
    text: str
    words: List[VerbatimWord] = field(default_factory=list)

    def get_start(self) -> float:
        return samples_to_seconds(self.start_ts)

    def get_end(self) -> float:
        return samples_to_seconds(self.end_ts)

    @classmethod
    def from_words(
        cls, words: List[VerbatimWord], speaker: str = None
    ) -> "VerbatimUtterance":
        start_ts = words[0].start_ts
        end_ts = words[-1].end_ts
        text = "".join([w.word for w in words])
        return cls(
            start_ts=start_ts, end_ts=end_ts, words=words, text=text, speaker=speaker
        )

from typing import List, Optional

from verbatim.transcript.words import Utterance, Word

from .writer import TranscriptWriter, TranscriptWriterConfig


class MultiTranscriptWriter(TranscriptWriter):
    def __init__(self, writers: Optional[List[TranscriptWriter]] = None):
        super().__init__(config=TranscriptWriterConfig())
        self.writers = []
        if writers is not None:
            self.writers += writers

    def get_extension(self):
        return ""

    def format_utterance(
        self,
        utterance: Utterance,
        unacknowledged_utterance: Optional[List[Utterance]] = None,
        unconfirmed_words: Optional[List[Word]] = None,
    ) -> bytes:
        for w in self.writers:
            w.format_utterance(
                utterance=utterance,
                unacknowledged_utterance=unacknowledged_utterance,
                unconfirmed_words=unconfirmed_words,
            )
        return b""

    def flush(self) -> bytes:
        for w in self.writers:
            w.flush()
        return b""

    def add_writer(self, writer: TranscriptWriter):
        self.writers.append(writer)

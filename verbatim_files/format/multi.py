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

    def open(self, path_no_ext: str):
        for w in self.writers:
            w.open(path_no_ext)

    def close(self):
        for w in self.writers:
            w.close()

    def write(
        self,
        utterance: Utterance,
        unacknowledged_utterance: Optional[List[Utterance]] = None,
        unconfirmed_words: Optional[List[Word]] = None,
    ):
        for w in self.writers:
            w.write(
                utterance=utterance,
                unacknowledged_utterance=unacknowledged_utterance,
                unconfirmed_words=unconfirmed_words,
            )

    def add_writer(self, writer: TranscriptWriter):
        self.writers.append(writer)

from typing import List

from .writer import TranscriptWriter, TranscriptWriterConfig
from ..words import VerbatimUtterance, VerbatimWord


class MultiTranscriptWriter(TranscriptWriter):
    def __init__(self, writers: List[TranscriptWriter] = None):
        super().__init__(config=TranscriptWriterConfig())
        self.writers = []
        if writers:
            self.writers += writers

    def get_extension(self):
        return ""

    def open(self, path_no_ext:str):
        for w in self.writers:
            w.open(path_no_ext)

    def close(self):
        for w in self.writers:
            w.close()

    def write(self,
              utterance:VerbatimUtterance,
              unacknowledged_utterance:List[VerbatimUtterance] = None,
              unconfirmed_words:List[VerbatimWord] = None):
        for w in self.writers:
            w.write(utterance=utterance, unacknowledged_utterance=unacknowledged_utterance, unconfirmed_words=unconfirmed_words)

    def add_writer(self, writer:TranscriptWriter):
        self.writers.append(writer)

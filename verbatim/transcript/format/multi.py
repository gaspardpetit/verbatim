from typing import List

from verbatim.transcript.format.writer import TranscriptWriter, TranscriptWriterConfig
from verbatim.transcript.transcript import Transcript
from verbatim.transcript.words import VerbatimUtterance


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

    def write(self, utterance:VerbatimUtterance):
        for w in self.writers:
            w.write(utterance=utterance)

    def add_writer(self, writer:TranscriptWriter):
        self.writers.append(writer)

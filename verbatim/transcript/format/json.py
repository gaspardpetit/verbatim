from typing import TextIO, List, Optional

from .writer import TranscriptWriter, TranscriptWriterConfig
from ..words import Utterance, Word


class TranscriptFormatter:
    def __init__(self):
        self.current_language = None
        self.current_speaker = None
        self.first_utterance = True

    def open(self, out: TextIO):
        out.write("[\n")

    def close(self, out: TextIO):
        out.write("\n]")

    def format_utterance(self, utterance: Utterance, out: TextIO):
        if self.first_utterance:
            self.first_utterance = False
        else:
            out.write(",\n")
        out.write("  {\n")
        out.write(f'    "start_sample": {utterance.start_ts}, "start_second": {utterance.start_ts / 16000:.2f},\n')
        out.write(f'    "end_sample": {utterance.end_ts}, "end_second": {utterance.end_ts / 16000:.2f},\n')
        out.write(f'    "speaker": "{utterance.speaker}",\n')
        out.write(f'    "text": "{utterance.text}",\n')
        out.write('    "words": [')
        first_word = True
        for w in utterance.words:
            if first_word:
                first_word = False
                out.write("\n")
            else:
                out.write(",\n")
            out.write(f'      {{ "text": "{w.word}", "lang": "{w.lang}", "prob": {w.probability:.4f} }}')
        out.write("\n    ]\n")
        out.write("  }")


class JsonTranscriptWriter(TranscriptWriter):
    out: TextIO

    def __init__(self, config: TranscriptWriterConfig):
        super().__init__(config)
        self.formatter: TranscriptFormatter = TranscriptFormatter()

    def open(self, path_no_ext: str):
        # pylint: disable=consider-using-with
        self.out = open(f"{path_no_ext}.json", "w", encoding="utf-8")
        self.formatter.open(out=self.out)

    def close(self):
        self.formatter.close(out=self.out)
        self.out.close()

    def write(
        self,
        utterance: Utterance,
        unacknowledged_utterance: Optional[List[Utterance]] = None,
        unconfirmed_words: Optional[List[Word]] = None,
    ):
        self.formatter.format_utterance(utterance=utterance, out=self.out)
        self.out.flush()

from typing import TextIO

from verbatim.transcript.format.writer import TranscriptWriter, TranscriptWriterConfig
from verbatim.transcript.formatting import format_milliseconds
from verbatim.transcript.words import VerbatimUtterance


class TranscriptFormatter:
    def __init__(self):
        self.current_language = None
        self.current_speaker = None
        self.first_utterance = True

    def open(self, out:TextIO):
        out.write("[\n")

    def close(self, out:TextIO):
        out.write("\n]")

    def format_utterance(self, utterance:VerbatimUtterance, out:TextIO):
        if self.first_utterance:
            self.first_utterance = False
        else:
            out.write(',\n')
        out.write('  {\n')
        out.write(f'    "start": {utterance.start_ts}, "start_ms": {utterance.start_ts * 1000 / 16000},\n')
        out.write(f'    "end": {utterance.end_ts}, "end_ms": {utterance.end_ts * 1000 / 16000},\n')
        out.write(f'    "speaker": "{utterance.speaker}",\n')
        out.write(f'    "words": [')
        first_word = True
        for w in utterance.words:
            if first_word:
                first_word = False
                out.write('\n')
            else:
                out.write(',\n')
            out.write(f'      {{ "text": "{w.word}", "lang": "{w.lang}", "prob": {w.probability} }}')
        out.write('    ]\n')
        out.write('  }')

class JsonTranscriptWriter(TranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig):
        super().__init__(config)
        self.formatter:TranscriptFormatter = TranscriptFormatter()
        self.out:TextIO = None

    def open(self, path_no_ext:str):
        self.out = open(f"{path_no_ext}.json", "w", encoding="utf-8")
        self.formatter.open(out=self.out)

    def close(self):
        self.formatter.close(out=self.out)
        self.out.close()

    def write(self, utterance:VerbatimUtterance):
        self.formatter.format_utterance(utterance=utterance, out=self.out)
        self.out.flush()
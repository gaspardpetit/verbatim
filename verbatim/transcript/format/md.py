from typing import TextIO, Union

from .writer import TranscriptWriter, TranscriptWriterConfig
from ..formatting import format_milliseconds
from ..words import VerbatimUtterance


def bold(text:str) -> str:
    return "**" + text + "**"

def ital(text:str) -> str:
    return "*" + text + "*"

class TranscriptFormatter:
    def __init__(self):
        self.current_language = None
        self.current_speaker = None

    def format_utterance(self, utterance:VerbatimUtterance, out:TextIO):
        line_text:str = ""
        line_header:str = ""
        line_header += f"[{format_milliseconds(utterance.start_ts * 1000 / 16000)}-{format_milliseconds(utterance.end_ts * 1000 / 16000)}]"
        if utterance.speaker != self.current_speaker:
            self.current_speaker = utterance.speaker
            line_header += f"[{utterance.speaker}]"
        for w in utterance.words:
            if w.lang != self.current_language:
                line_text += ital(f"[{w.lang}]")
                self.current_language = w.lang
            line_text += w.word
        line_text += '\n\n'
        out.write(bold(line_header + ":") + line_text)

class MarkdownTranscriptWriter(TranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig):
        super().__init__(config)
        self.formatter:TranscriptFormatter = TranscriptFormatter()
        self.out:Union[None,TextIO] = None

    def open(self, path_no_ext:str):
        # pylint: disable=consider-using-with
        self.out = open(f"{path_no_ext}.md", "w", encoding="utf-8")

    def close(self):
        self.out.close()

    def write(self, utterance:VerbatimUtterance):
        self.formatter.format_utterance(utterance=utterance, out=self.out)
        self.out.flush()

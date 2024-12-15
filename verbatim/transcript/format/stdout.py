import sys

from verbatim.transcript.format.txt import TextIOTranscriptWriter, COLORSCHEME_ACKNOWLEDGED, COLORSCHEME_NONE
from verbatim.transcript.format.writer import TranscriptWriterConfig


class StdoutTranscriptWriter(TextIOTranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig, with_colours:bool = True):
        super().__init__(config=config, out=sys.stdout,
                         colours=COLORSCHEME_ACKNOWLEDGED if with_colours else COLORSCHEME_NONE)

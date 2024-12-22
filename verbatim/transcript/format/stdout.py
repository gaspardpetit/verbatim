import sys

from .txt import (
    TextIOTranscriptWriter,
    COLORSCHEME_ACKNOWLEDGED,
    COLORSCHEME_UNACKNOWLEDGED,
    COLORSCHEME_UNCONFIRMED,
    COLORSCHEME_NONE
)
from .writer import TranscriptWriterConfig


class StdoutTranscriptWriter(TextIOTranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig, with_colours:bool = True):
        super().__init__(
            config=config, out=sys.stdout,
            acknowledged_colours=COLORSCHEME_ACKNOWLEDGED if with_colours else COLORSCHEME_NONE,
            unacknowledged_colours=COLORSCHEME_UNACKNOWLEDGED if with_colours else COLORSCHEME_NONE,
            unconfirmed_colors=COLORSCHEME_UNCONFIRMED if with_colours else COLORSCHEME_NONE,
            print_unacknowledged=config.verbose
        )

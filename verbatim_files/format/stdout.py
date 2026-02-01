from .txt import (
    COLORSCHEME_ACKNOWLEDGED,
    COLORSCHEME_NONE,
    COLORSCHEME_UNACKNOWLEDGED,
    COLORSCHEME_UNCONFIRMED,
    TextIOTranscriptWriter,
)
from .writer import TranscriptWriterConfig


class StdoutTranscriptWriter(TextIOTranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig, with_colours: bool = True):
        super().__init__(
            config=config,
            acknowledged_colours=COLORSCHEME_ACKNOWLEDGED if with_colours else COLORSCHEME_NONE,
            unacknowledged_colours=COLORSCHEME_UNACKNOWLEDGED if with_colours else COLORSCHEME_NONE,
            unconfirmed_colors=COLORSCHEME_UNCONFIRMED if with_colours else COLORSCHEME_NONE,
            print_unacknowledged=config.verbose,
        )

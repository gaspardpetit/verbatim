from dataclasses import dataclass
from typing import TextIO, Union, List

from colorama import Fore, Style

from .writer import TranscriptWriter, TranscriptWriterConfig
from ..formatting import format_milliseconds
from ..words import Utterance, Word


@dataclass
class ColorScheme:
    color_timestamp: str = Fore.CYAN
    color_speaker: str = Fore.BLUE
    color_language: str = Fore.YELLOW
    color_text: str = Fore.GREEN
    color_reset: str = Style.RESET_ALL


COLORSCHEME_ACKNOWLEDGED = ColorScheme(
    color_timestamp=Fore.LIGHTCYAN_EX,
    color_speaker=Fore.LIGHTBLUE_EX,
    color_language=Fore.LIGHTYELLOW_EX,
    color_text=Fore.LIGHTGREEN_EX,
    color_reset=Style.RESET_ALL,
)

COLORSCHEME_UNACKNOWLEDGED = ColorScheme(
    color_timestamp=Fore.CYAN,
    color_speaker=Fore.BLUE,
    color_language=Fore.YELLOW,
    color_text=Fore.GREEN,
    color_reset=Style.RESET_ALL,
)

COLORSCHEME_UNCONFIRMED = ColorScheme(
    color_timestamp=Fore.CYAN,
    color_speaker=Fore.BLUE,
    color_language=Fore.YELLOW,
    color_text=Fore.LIGHTBLACK_EX,
    color_reset=Style.RESET_ALL,
)

COLORSCHEME_NONE = ColorScheme(
    color_timestamp="",
    color_speaker="",
    color_language="",
    color_text="",
    color_reset="",
)


class TranscriptFormatter:
    def __init__(self):
        self.current_language = None
        self.current_speaker = None

    def format_utterance(self, utterance: Utterance, out: TextIO, colours: ColorScheme):
        line: str = ""
        line += colours.color_timestamp
        line += f"[{format_milliseconds(utterance.start_ts * 1000 / 16000)}-{format_milliseconds(utterance.end_ts * 1000 / 16000)}]"
        if utterance.speaker != self.current_speaker:
            self.current_speaker = utterance.speaker
            line += colours.color_speaker
            line += f"[{utterance.speaker}]"
        line += colours.color_text
        for w in utterance.words:
            if w.lang != self.current_language:
                line += colours.color_language
                line += f"[{w.lang}]"
                line += colours.color_text
                self.current_language = w.lang
            line += w.word
        line += colours.color_reset
        line += "\n"
        out.write(line)


class TextIOTranscriptWriter(TranscriptWriter):
    def __init__(
        self,
        *,
        config: TranscriptWriterConfig,
        out: Union[None, TextIO],
        acknowledged_colours=COLORSCHEME_ACKNOWLEDGED,
        unacknowledged_colours=COLORSCHEME_UNACKNOWLEDGED,
        unconfirmed_colors=COLORSCHEME_UNCONFIRMED,
        print_unacknowledged: bool = False,
    ):
        super().__init__(config)
        self.formatter: TranscriptFormatter = TranscriptFormatter()
        self.out: TextIO = out
        self.acknowledged_colours = acknowledged_colours
        self.unacknowledged_colours = unacknowledged_colours
        self.unconfirmed_colors = unconfirmed_colors
        self.print_unacknowledged = print_unacknowledged

    def open(self, path_no_ext: str):
        pass

    def close(self):
        pass

    def write(
        self,
        utterance: Utterance,
        unacknowledged_utterance: List[Utterance] = None,
        unconfirmed_words: List[Word] = None,
    ):
        self.formatter.format_utterance(utterance=utterance, out=self.out, colours=self.acknowledged_colours)
        if self.print_unacknowledged:
            if unacknowledged_utterance:
                for unack in unacknowledged_utterance:
                    self.formatter.format_utterance(utterance=unack, out=self.out, colours=self.unconfirmed_colors)
            if unconfirmed_words and len(unconfirmed_words) > 0:
                self.formatter.format_utterance(
                    utterance=Utterance.from_words(utterance_id="", words=unconfirmed_words),
                    out=self.out,
                    colours=self.unconfirmed_colors,
                )
        self.out.flush()


class TextTranscriptWriter(TextIOTranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig):
        super().__init__(
            config=config,
            out=None,
            acknowledged_colours=COLORSCHEME_NONE,
            unacknowledged_colours=COLORSCHEME_NONE,
            unconfirmed_colors=COLORSCHEME_NONE,
        )

    def open(self, path_no_ext: str):
        # pylint: disable=consider-using-with
        self.out = open(f"{path_no_ext}.txt", "w", encoding="utf-8")

    def close(self):
        self.out.close()

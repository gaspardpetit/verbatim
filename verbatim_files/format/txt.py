from dataclasses import dataclass
from typing import List, Optional, TextIO

import numpy as np
from colorama import Fore, Style

from verbatim.transcript.words import Utterance, Word
from verbatim_diarization import UNKNOWN_SPEAKER

from ..formatting import format_milliseconds
from .writer import (
    LanguageStyle,
    ProbabilityStyle,
    SpeakerStyle,
    TimestampStyle,
    TranscriptWriter,
    TranscriptWriterConfig,
)


@dataclass
class ColorScheme:
    color_timestamp: str = Fore.CYAN
    color_speaker: str = Fore.BLUE
    color_language: str = Fore.YELLOW
    color_text: str = Fore.GREEN
    color_text_lowconfidence: str = Fore.LIGHTYELLOW_EX
    color_text_verylowconfidence: str = Fore.LIGHTRED_EX
    color_reset: str = Style.RESET_ALL


COLORSCHEME_ACKNOWLEDGED = ColorScheme(
    color_timestamp=Fore.LIGHTCYAN_EX,
    color_speaker=Fore.LIGHTBLUE_EX,
    color_language=Fore.LIGHTYELLOW_EX,
    color_text=Fore.LIGHTGREEN_EX,
    color_text_lowconfidence=Fore.LIGHTYELLOW_EX,
    color_text_verylowconfidence=Fore.LIGHTRED_EX,
    color_reset=Style.RESET_ALL,
)

COLORSCHEME_UNACKNOWLEDGED = ColorScheme(
    color_timestamp=Fore.CYAN,
    color_speaker=Fore.BLUE,
    color_language=Fore.YELLOW,
    color_text=Fore.GREEN,
    color_text_lowconfidence=Fore.LIGHTYELLOW_EX,
    color_text_verylowconfidence=Fore.LIGHTRED_EX,
    color_reset=Style.RESET_ALL,
)

COLORSCHEME_UNCONFIRMED = ColorScheme(
    color_timestamp=Fore.CYAN,
    color_speaker=Fore.BLUE,
    color_language=Fore.YELLOW,
    color_text=Fore.LIGHTBLACK_EX,
    color_text_lowconfidence=Fore.LIGHTBLACK_EX,
    color_text_verylowconfidence=Fore.LIGHTBLACK_EX,
    color_reset=Style.RESET_ALL,
)

COLORSCHEME_NONE = ColorScheme(
    color_timestamp="",
    color_speaker="",
    color_language="",
    color_text="",
    color_reset="",
    color_text_lowconfidence="",
    color_text_verylowconfidence="",
)


class TranscriptFormatter:
    def __init__(
        self,
        speaker_style: SpeakerStyle,
        timestamp_style: TimestampStyle,
        probability_style: ProbabilityStyle,
        language_style: LanguageStyle,
    ):
        self.current_language = None
        self.current_speaker = None
        self.speaker_style = speaker_style
        self.timestamp_style = timestamp_style
        self.probability_style = probability_style
        self.language_style = language_style
        self.last_ts: int = 0

    def _format_timestamp(self, out: TextIO, start_ts: int, end_ts: int, colours: ColorScheme):
        if self.timestamp_style == TimestampStyle.none:
            pass

        elif self.timestamp_style == TimestampStyle.minute:
            if self.last_ts is None or start_ts - self.last_ts >= 60 * 16000:
                self.last_ts = ((start_ts // 16000) // 60) * 60 * 16000
                out.write("\n")
                out.write(colours.color_timestamp)
                out.write(f"[{format_milliseconds(start_ts * 1000 / 16000)}]")
                out.write(colours.color_reset)
                out.write("\n\n")

        elif self.timestamp_style == TimestampStyle.start:
            out.write(colours.color_timestamp)
            out.write(f"[{format_milliseconds(start_ts * 1000 / 16000)}]:")
            out.write(colours.color_reset)

        elif self.timestamp_style == TimestampStyle.range:
            out.write(colours.color_timestamp)
            out.write(f"[{format_milliseconds(start_ts * 1000 / 16000)}-{format_milliseconds(end_ts * 1000 / 16000)}]:")
            out.write(colours.color_reset)

    def _format_speaker(self, out: TextIO, speaker: str, colours: ColorScheme):
        if self.speaker_style == SpeakerStyle.none:
            return

        elif self.speaker_style == SpeakerStyle.change:
            if speaker != self.current_speaker:
                self.current_speaker = speaker
                out.write(colours.color_speaker)
                out.write(f"[{speaker}]")
                out.write(colours.color_reset)

        elif self.speaker_style == SpeakerStyle.always:
            out.write(colours.color_speaker)
            out.write(f"[{speaker}]")
            out.write(colours.color_reset)

    def _format_language(self, out: TextIO, language: str, first_word: bool, colours: ColorScheme):
        if self.language_style == LanguageStyle.none:
            pass
        elif self.language_style == LanguageStyle.change:
            if language != self.current_language:
                self.current_language = language
                out.write(colours.color_language)
                out.write(f"[{language}]")
                out.write(colours.color_reset)
        elif self.language_style == LanguageStyle.always:
            if first_word or language != self.current_language:
                self.current_language = language
                out.write(colours.color_language)
                out.write(f"[{language}]")
                out.write(colours.color_reset)

    def _format_word_with_probability(
        self,
        *,
        out: TextIO,
        word: str,
        probability: float,
        utterance_probability: float,
        colours: ColorScheme,
    ):
        # pylint: disable=too-many-boolean-expressions
        if (
            self.probability_style == ProbabilityStyle.word
            and probability < 0.90 / 2
            or self.probability_style == ProbabilityStyle.word_75
            and probability < 0.75 / 2
            or self.probability_style == ProbabilityStyle.word_50
            and probability < 0.50 / 2
            or self.probability_style == ProbabilityStyle.word_25
            and probability < 0.25 / 2
        ):
            out.write(colours.color_text_verylowconfidence)
            out.write(word)
            out.write(colours.color_reset)
        elif (
            self.probability_style == ProbabilityStyle.word
            and probability < 0.90
            or self.probability_style == ProbabilityStyle.word_75
            and probability < 0.75
            or self.probability_style == ProbabilityStyle.word_50
            and probability < 0.50
            or self.probability_style == ProbabilityStyle.word_25
            and probability < 0.25
        ):
            out.write(colours.color_text_lowconfidence)
            out.write(word)
            out.write(colours.color_reset)
        elif (
            self.probability_style == ProbabilityStyle.line
            and utterance_probability < 0.90 / 2
            or self.probability_style == ProbabilityStyle.line_75
            and utterance_probability < 0.75 / 2
            or self.probability_style == ProbabilityStyle.line_50
            and utterance_probability < 0.50 / 2
            or self.probability_style == ProbabilityStyle.line_25
            and utterance_probability < 0.25 / 2
        ):
            out.write(colours.color_text_verylowconfidence)
            out.write(word)
            out.write(colours.color_reset)
        elif (
            self.probability_style == ProbabilityStyle.line
            and utterance_probability < 0.90
            or self.probability_style == ProbabilityStyle.line_75
            and utterance_probability < 0.75
            or self.probability_style == ProbabilityStyle.line_50
            and utterance_probability < 0.50
            or self.probability_style == ProbabilityStyle.line_25
            and utterance_probability < 0.25
        ):
            out.write(colours.color_text_lowconfidence)
            out.write(word)
            out.write(colours.color_reset)
        else:
            out.write(colours.color_text)
            out.write(word)
            out.write(colours.color_reset)

    def format_utterance(self, utterance: Utterance, out: TextIO, colours: ColorScheme):
        self._format_timestamp(out=out, start_ts=utterance.start_ts, end_ts=utterance.end_ts, colours=colours)
        self._format_speaker(out=out, speaker=utterance.speaker or UNKNOWN_SPEAKER, colours=colours)

        percentile_25 = float(np.percentile([w.probability for w in utterance.words], 25))

        # pylint: disable=superfluous-parens
        for i, w in enumerate(utterance.words):
            self._format_language(out=out, language=w.lang, first_word=(i == 0), colours=colours)
            self._format_word_with_probability(out=out, word=w.word, probability=w.probability, utterance_probability=percentile_25, colours=colours)
        out.write("\n")


class TextIOTranscriptWriter(TranscriptWriter):
    out: TextIO

    def __init__(
        self,
        *,
        config: TranscriptWriterConfig,
        acknowledged_colours=COLORSCHEME_ACKNOWLEDGED,
        unacknowledged_colours=COLORSCHEME_UNACKNOWLEDGED,
        unconfirmed_colors=COLORSCHEME_UNCONFIRMED,
        print_unacknowledged: bool = False,
    ):
        super().__init__(config)
        self.formatter: TranscriptFormatter = TranscriptFormatter(
            language_style=config.language_style,
            probability_style=config.probability_style,
            speaker_style=config.speaker_style,
            timestamp_style=config.timestamp_style,
        )
        self.acknowledged_colours = acknowledged_colours
        self.unacknowledged_colours = unacknowledged_colours
        self.unconfirmed_colors = unconfirmed_colors
        self.print_unacknowledged = print_unacknowledged

    def _set_textio(self, out: TextIO):
        self.out = out

    def open(self, path_no_ext: str):
        pass

    def close(self):
        pass

    def write(
        self,
        utterance: Utterance,
        unacknowledged_utterance: Optional[List[Utterance]] = None,
        unconfirmed_words: Optional[List[Word]] = None,
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
            acknowledged_colours=COLORSCHEME_NONE,
            unacknowledged_colours=COLORSCHEME_NONE,
            unconfirmed_colors=COLORSCHEME_NONE,
        )

    def open(self, path_no_ext: str):
        # pylint: disable=consider-using-with
        self._set_textio(open(f"{path_no_ext}.txt", "w", encoding="utf-8"))

    def close(self):
        self.out.close()

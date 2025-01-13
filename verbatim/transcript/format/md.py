from typing import TextIO, Union, List, Optional
from enum import Enum
import numpy as np

from .writer import (
    TranscriptWriter,
    TranscriptWriterConfig,
    SpeakerStyle,
    TimestampStyle,
    ProbabilityStyle,
    LanguageStyle,
)
from ..formatting import format_milliseconds
from ..words import Utterance, Word
from ...voices.diarization import UNKNOWN_SPEAKER


class Style(Enum):
    NONE = 0
    ITALIC = 1
    BOLD = 2
    UNDERLINE = 3


class MarkdownText:
    def __init__(self):
        self.text = []
        self.style: List[Style] = []

    def __str__(self) -> str:
        text = "".join(self.text)
        for style in reversed(self.style):
            if style == Style.BOLD:
                text += "</b>"
            if style == Style.ITALIC:
                text += "</i>"
            if style == Style.UNDERLINE:
                text += "</u>"
        return text

    def _set_style(self, style: Union[Style, List[Style], None] = None):
        def _close_style(style: Style):
            if style == Style.BOLD:
                self.text.append("</b>")
            elif style == Style.ITALIC:
                self.text.append("</i>")
            elif style == Style.UNDERLINE:
                self.text.append("</u>")

        def _open_style(style: Style):
            if style == Style.BOLD:
                self.text.append("<b>")
            elif style == Style.ITALIC:
                self.text.append("<i>")
            elif style == Style.UNDERLINE:
                self.text.append("<u>")

        def _close_styles(styles: List[Style]):
            for st in reversed(styles):
                _close_style(st)

        # Case 1: style is None, an empty list, or Style.NONE
        # => close all currently open styles
        if style is None or style == [] or style == Style.NONE:
            _close_styles(self.style)
            self.style = []
            return

        elif isinstance(style, Style):
            if len(self.style) > 0 and self.style[0] == style:
                _close_styles(self.style[1:])
            else:
                _close_styles(self.style)
                _open_style(style)
            self.style = [style]

        elif isinstance(style, list):
            prefix_index = 0
            for active_style in self.style:
                if active_style in style:
                    prefix_index += 1
                else:
                    break

            _close_styles(self.style[prefix_index:])
            self.style = self.style[: prefix_index + 1]
            for reqested_style in style:
                if reqested_style not in self.style:
                    _open_style(reqested_style)
                    self.style.append(reqested_style)

    def append(self, text: str, style: Union[Style, List[Style], None] = None):
        self._set_style(style)
        self.text.append(text)

    def bold(self, text: str):
        self.append(text, Style.BOLD)

    def underline(self, text: str):
        self.append(text, Style.UNDERLINE)

    def italic(self, text: str):
        self.append(text, Style.ITALIC)


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

    def _format_timestamp(self, md: MarkdownText, start_ts: int, end_ts: int):
        if self.timestamp_style == TimestampStyle.none:
            pass

        elif self.timestamp_style == TimestampStyle.minute:
            if self.last_ts is None or start_ts - self.last_ts >= 60 * 16000:
                self.last_ts = ((start_ts // 16000) // 60) * 60 * 16000
                md.bold(f"[{format_milliseconds(start_ts * 1000 / 16000)}]")
                md.append("\n\n")

        elif self.timestamp_style == TimestampStyle.start:
            md.bold(f"[{format_milliseconds(start_ts * 1000 / 16000)}]:")

        elif self.timestamp_style == TimestampStyle.range:
            md.bold(f"[{format_milliseconds(start_ts * 1000 / 16000)}-{format_milliseconds(end_ts * 1000 / 16000)}]:")

    def _format_speaker(self, md: MarkdownText, speaker: str):
        if self.speaker_style == SpeakerStyle.none:
            return

        elif self.speaker_style == SpeakerStyle.change:
            if speaker != self.current_speaker:
                self.current_speaker = speaker
                md.bold(f"[{speaker}]")

        elif self.speaker_style == SpeakerStyle.always:
            md.bold(f"[{speaker}]")

    def _format_language(self, md: MarkdownText, language: str, first_word: bool):
        if self.language_style == LanguageStyle.none:
            pass
        elif self.language_style == LanguageStyle.change:
            if language != self.current_language:
                self.current_language = language
                md.bold(f"[{language}]")
        elif self.language_style == LanguageStyle.always:
            if first_word or language != self.current_language:
                self.current_language = language
                md.bold(f"[{language}]")

    def _format_word_with_probability(
        self,
        md: MarkdownText,
        word: str,
        probability: float,
        utterance_probability: float,
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
            md.underline(word)
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
            md.italic(word)
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
            md.underline(word)
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
            md.italic(word)
        else:
            md.append(word)

    def format_utterance(self, utterance: Utterance, out: TextIO):
        md: MarkdownText = MarkdownText()
        self._format_timestamp(md=md, start_ts=utterance.start_ts, end_ts=utterance.end_ts)
        self._format_speaker(md=md, speaker=utterance.speaker or UNKNOWN_SPEAKER)

        percentile_25 = float(np.percentile([w.probability for w in utterance.words], 25))

        # pylint: disable=superfluous-parens
        for i, w in enumerate(utterance.words):
            self._format_language(md=md, language=w.lang, first_word=(i == 0))
            self._format_word_with_probability(
                md=md,
                word=w.word,
                probability=w.probability,
                utterance_probability=percentile_25,
            )
        md.append("\n\n")
        out.write(str(md))


class MarkdownTranscriptWriter(TranscriptWriter):
    out: TextIO

    def __init__(self, config: TranscriptWriterConfig):
        super().__init__(config)
        self.formatter: TranscriptFormatter = TranscriptFormatter(
            language_style=config.language_style,
            timestamp_style=config.timestamp_style,
            speaker_style=config.speaker_style,
            probability_style=config.probability_style,
        )

    def open(self, path_no_ext: str):
        # pylint: disable=consider-using-with
        self.out = open(f"{path_no_ext}.md", "w", encoding="utf-8")

    def close(self):
        self.out.close()

    def write(
        self,
        utterance: Utterance,
        unacknowledged_utterance: Optional[List[Utterance]] = None,
        unconfirmed_words: Optional[List[Word]] = None,
    ):
        self.formatter.format_utterance(utterance=utterance, out=self.out)
        self.out.flush()

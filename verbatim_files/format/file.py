from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Iterable, Optional

from .writer import TranscriptWriter


@dataclass
class FileFormatter:
    writer: TranscriptWriter
    output_path: Optional[str] = None
    output: Optional[BinaryIO] = None
    close_output: bool = True

    def open(self) -> None:
        if self.output is None:
            if self.output_path is None:
                raise ValueError("output_path or output must be provided")
            self.output = open(self.output_path, "wb")
            self.close_output = True
        self._write(self.writer.format_start())

    def write(
        self,
        utterance,
        unacknowledged_utterance=None,
        unconfirmed_words=None,
    ) -> None:
        self._write(
            self.writer.format_utterance(
                utterance=utterance,
                unacknowledged_utterance=unacknowledged_utterance,
                unconfirmed_words=unconfirmed_words,
            )
        )

    def close(self) -> None:
        self._write(self.writer.flush())
        self._write(self.writer.format_end())
        if self.output is not None and self.close_output:
            self.output.close()
        if self.output_path and hasattr(self.writer, "post_close"):
            self.writer.post_close(self.output_path)

    def _write(self, data: bytes) -> None:
        if not data:
            return
        if self.output is None:
            raise ValueError("Formatter output stream is not open")
        self.output.write(data)
        if hasattr(self.output, "flush"):
            self.output.flush()


@dataclass
class MultiFileFormatter:
    formatters: Iterable[FileFormatter]

    def open(self) -> None:
        for formatter in self.formatters:
            formatter.open()

    def write(
        self,
        utterance,
        unacknowledged_utterance=None,
        unconfirmed_words=None,
    ) -> None:
        for formatter in self.formatters:
            formatter.write(
                utterance=utterance,
                unacknowledged_utterance=unacknowledged_utterance,
                unconfirmed_words=unconfirmed_words,
            )

    def close(self) -> None:
        for formatter in self.formatters:
            formatter.close()

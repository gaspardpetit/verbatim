import sys
from typing import BinaryIO, Optional

from verbatim.logging_utils import get_status_logger
from verbatim.status_types import StatusUpdate
from verbatim_files.format.stdout import StdoutTranscriptWriter
from verbatim_files.format.writer import TranscriptWriterConfig


class SimpleProgressHook:
    def __init__(
        self,
        *,
        config: TranscriptWriterConfig,
        with_colours: bool = True,
        output: Optional[BinaryIO] = None,
    ):
        self._writer = StdoutTranscriptWriter(config=config, with_colours=with_colours)
        self._output = output or sys.stdout.buffer
        self._last_progress_log: Optional[float] = None
        self._last_state: Optional[str] = None
        self._status_log = get_status_logger()

    def __call__(self, update: StatusUpdate) -> None:
        if update.state and update.state != "utterance" and update.state != self._last_state:
            self._status_log.info("State: %s", update.state)
            self._last_state = update.state
        if update.progress and update.progress.units == "seconds" and update.state == "transcribing":
            current = update.progress.current
            if self._last_progress_log is None or current - self._last_progress_log >= 10:
                finish = update.progress.finish
                if finish:
                    self._status_log.info("Transcribing progress: %.1fs / %.1fs", current, finish)
                else:
                    self._status_log.info("Transcribing progress: %.1fs", current)
                self._last_progress_log = current
        elif update.progress:
            label = update.state or "progress"
            finish = update.progress.finish
            if finish is not None:
                self._status_log.info("%s progress: %.1f / %.1f %s", label, update.progress.current, finish, update.progress.units)
            else:
                self._status_log.info("%s progress: %.1f %s", label, update.progress.current, update.progress.units)
        if update.utterance is None:
            return
        payload = self._writer.format_utterance(
            utterance=update.utterance,
            unacknowledged_utterance=update.unacknowledged_utterances,
            unconfirmed_words=update.unconfirmed_words,
        )
        if not payload:
            return
        self._output.write(payload)
        if hasattr(self._output, "flush"):
            self._output.flush()

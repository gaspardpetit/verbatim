from __future__ import annotations

import platform
import sys
from time import perf_counter
from typing import TYPE_CHECKING, Optional

from verbatim.logging_utils import get_status_logger
from verbatim_transcript import TranscriberProtocol

from .transcript.sentences import SentenceTokenizer

if TYPE_CHECKING:
    from .voices.silences import VoiceActivityDetection

STATUS_LOG = get_status_logger()


class Models:
    vad: "VoiceActivityDetection"
    sentence_tokenizer: SentenceTokenizer

    def __init__(
        self,
        device: str,
        whisper_model_size: str = "large-v3",
        stream: bool = False,
        transcriber: Optional[TranscriberProtocol] = None,
    ):
        self._transcriber: Optional[TranscriberProtocol] = transcriber
        self._device = device
        self._whisper_model_size = whisper_model_size

        if transcriber is not None:
            STATUS_LOG.info("Using injected transcriber implementation.")
        else:
            STATUS_LOG.info("Transcriber will be lazy-loaded on first use (device=%s, size=%s).", device, whisper_model_size)

        STATUS_LOG.info("Lazy-loading Silero VAD model.")
        vad_start = perf_counter()
        from .voices.silences import SileroVoiceActivityDetection  # pylint: disable=import-outside-toplevel

        self.vad: "VoiceActivityDetection" = SileroVoiceActivityDetection()
        STATUS_LOG.info("Silero VAD loaded in %.2fs", perf_counter() - vad_start)

        STATUS_LOG.info("Initializing Sentence Tokenizer.")
        if stream:
            from .transcript.sentences import FastSentenceTokenizer  # pylint: disable=import-outside-toplevel

            self.sentence_tokenizer = FastSentenceTokenizer()
        else:
            STATUS_LOG.info("Lazy-loading SaT sentence tokenizer.")
            sat_start = perf_counter()
            from .transcript.sentences import SaTSentenceTokenizer  # pylint: disable=import-outside-toplevel

            self.sentence_tokenizer = SaTSentenceTokenizer(device)
            STATUS_LOG.info("SaT sentence tokenizer loaded in %.2fs", perf_counter() - sat_start)

    @property
    def transcriber(self) -> TranscriberProtocol:
        if self._transcriber is None:
            transcriber_start = perf_counter()
            self._transcriber = self._build_transcriber()
            STATUS_LOG.info("Transcriber ready in %.2fs", perf_counter() - transcriber_start)
        return self._transcriber

    def _build_transcriber(self) -> TranscriberProtocol:
        # pylint: disable=import-outside-toplevel
        if sys.platform == "darwin":
            # If this is an Apple Silicon device, use the MLX Whisper transcriber
            if platform.processor() == "arm":
                STATUS_LOG.info("Using WhisperMLX transcriber on Apple Silicon")
                from .voices.transcribe.whispermlx import WhisperMlxTranscriber  # pylint: disable=import-outside-toplevel

                return WhisperMlxTranscriber(model_size_or_path=self._whisper_model_size)

            # Use WhisperCPP on Mac by default
            STATUS_LOG.info("Using WhisperCPP transcriber on Mac OS X")
            from .voices.transcribe.whispercpp import WhisperCppTranscriber  # pylint: disable=import-outside-toplevel

            return WhisperCppTranscriber(model_size_or_path=self._whisper_model_size, device=self._device)

        STATUS_LOG.info("Using 'faster-whisper' transcriber.")
        from .voices.transcribe.faster_whisper import FasterWhisperTranscriber  # pylint: disable=import-outside-toplevel

        return FasterWhisperTranscriber(model_size_or_path=self._whisper_model_size, device=self._device)

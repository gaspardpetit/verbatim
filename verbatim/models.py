import logging
import platform
import sys
from typing import Optional

from verbatim_transcript import TranscriberProtocol

from .transcript.sentences import SentenceTokenizer
from .voices.silences import SileroVoiceActivityDetection, VoiceActivityDetection

# Configure logger
LOG = logging.getLogger(__name__)


class Models:
    transcriber: TranscriberProtocol
    vad: VoiceActivityDetection
    sentence_tokenizer: SentenceTokenizer

    def __init__(
        self,
        device: str,
        whisper_model_size: str = "large-v3",
        stream: bool = False,
        transcriber: Optional[TranscriberProtocol] = None,
    ):
        # pylint: disable=import-outside-toplevel
        LOG.info("Initializing WhisperModel and audio stream.")

        if transcriber is not None:
            LOG.info("Using injected transcriber implementation.")
            self.transcriber = transcriber  # type: ignore[assignment]
        elif sys.platform == "darwin":
            # If there this is an Apple Silicon device, use the MLX Whisper transcriber
            if platform.processor() == "arm":
                LOG.info("Using WhisperMLX transcriber on Apple Silicon")
                from .voices.transcribe.whispermlx import WhisperMlxTranscriber

                self.transcriber = WhisperMlxTranscriber(model_size_or_path=whisper_model_size)
            else:
                # Use WhisperCPP on Mac by default
                LOG.info("Using WhisperCPP transcriber on Mac OS X")
                from .voices.transcribe.whispercpp import WhisperCppTranscriber

                self.transcriber = WhisperCppTranscriber(model_size_or_path=whisper_model_size, device=device)
        else:
            LOG.info("Using 'faster-whisper' transcriber.")
            from .voices.transcribe.faster_whisper import FasterWhisperTranscriber

            self.transcriber = FasterWhisperTranscriber(model_size_or_path=whisper_model_size, device=device)

        LOG.info("Initializing Silero VAD model.")
        self.vad: VoiceActivityDetection = SileroVoiceActivityDetection()

        LOG.info("Initializing Sentence Tokenizer.")
        if stream:
            from .transcript.sentences import FastSentenceTokenizer

            self.sentence_tokenizer = FastSentenceTokenizer()
        else:
            from .transcript.sentences import SaTSentenceTokenizer

            self.sentence_tokenizer = SaTSentenceTokenizer(device)

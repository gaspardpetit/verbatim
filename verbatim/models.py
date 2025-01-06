import logging
import sys

from .voices.transcribe.transcribe import Transcriber
from .voices.silences import SileroVoiceActivityDetection, VoiceActivityDetection
from .transcript.sentences import SentenceTokenizer

# Configure logger
LOG = logging.getLogger(__name__)


class Models:
    transcriber: Transcriber = None
    vad: VoiceActivityDetection = None
    sentence_tokenizer: SentenceTokenizer = None

    def __init__(
        self, device: str, whisper_model_size: str = "large-v3", stream: bool = False
    ):
        LOG.info("Initializing WhisperModel and audio stream.")

        if sys.platform == "darwin":
            # Use WhisperCPP on Mac by default
            LOG.info("Using WhisperCPP transcriber on Mac OS X")
            # pylint: disable=import-outside-toplevel
            from .voices.transcribe.whispercpp import WhisperCppTranscriber

            self.transcriber = WhisperCppTranscriber(
                model_size_or_path=whisper_model_size, device=device
            )
        else:
            LOG.info("Using 'faster-whisper' transcriber.")
            # pylint: disable=import-outside-toplevel
            from .voices.transcribe.faster_whisper import FasterWhisperTranscriber

            self.transcriber = FasterWhisperTranscriber(
                model_size_or_path=whisper_model_size, device=device
            )

        LOG.info("Initializing Silero VAD model.")
        self.vad: VoiceActivityDetection = SileroVoiceActivityDetection()

        LOG.info("Initializing Sentence Tokenizer.")
        if stream:
            from .transcript.sentences import FastSentenceTokenizer

            self.sentence_tokenizer = FastSentenceTokenizer()
        else:
            from .transcript.sentences import SaTSentenceTokenizer

            self.sentence_tokenizer = SaTSentenceTokenizer(device)

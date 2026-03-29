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
        *,
        device: str,
        whisper_model_size: str = "large-v3",
        voxtral_model_size: str = "mistralai/Voxtral-Mini-3B-2507",
        voxtral_dtype: str = "auto",
        voxtral_max_new_tokens: int = 256,
        stream: bool = False,
        transcriber: Optional[TranscriberProtocol] = None,
        transcriber_backend: str = "auto",
        qwen_asr_model_size: str = "Qwen/Qwen3-ASR-1.7B",
        qwen_aligner_model_size: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        qwen_dtype: str = "auto",
        qwen_max_inference_batch_size: int = 1,
        qwen_max_new_tokens: int = 256,
    ):
        self._transcriber: Optional[TranscriberProtocol] = transcriber
        self._device = device
        self._whisper_model_size = whisper_model_size
        self._voxtral_model_size = voxtral_model_size
        self._voxtral_dtype = voxtral_dtype
        self._voxtral_max_new_tokens = voxtral_max_new_tokens
        self._config_transcriber_backend = transcriber_backend
        self._qwen_asr_model_size = qwen_asr_model_size
        self._qwen_aligner_model_size = qwen_aligner_model_size
        self._qwen_dtype = qwen_dtype
        self._qwen_max_inference_batch_size = qwen_max_inference_batch_size
        self._qwen_max_new_tokens = qwen_max_new_tokens

        if transcriber is not None:
            STATUS_LOG.info("Using injected transcriber implementation.")
        else:
            STATUS_LOG.info(
                "Transcriber will be lazy-loaded on first use (backend=%s, device=%s, size=%s).",
                transcriber_backend,
                device,
                whisper_model_size,
            )

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
        backend = (getattr(self, "_config_transcriber_backend", "auto") or "auto").lower()
        if backend in ("qwen", "qwen-asr"):
            STATUS_LOG.info("Using Qwen3-ASR transcriber.")
            from .voices.transcribe.qwen_asr import QwenAsrTranscriber  # pylint: disable=import-outside-toplevel

            return QwenAsrTranscriber(
                model_size_or_path=self._qwen_asr_model_size,
                aligner_model_size_or_path=self._qwen_aligner_model_size,
                device=self._device,
                dtype=self._qwen_dtype,
                max_inference_batch_size=self._qwen_max_inference_batch_size,
                max_new_tokens=self._qwen_max_new_tokens,
            )

        if backend == "voxtral":
            STATUS_LOG.info("Using Voxtral transcriber with Qwen forced alignment.")
            from .voices.transcribe.voxtral import VoxtralTranscriber  # pylint: disable=import-outside-toplevel

            return VoxtralTranscriber(
                model_size_or_path=self._voxtral_model_size,
                aligner_model_size_or_path=self._qwen_aligner_model_size,
                device=self._device,
                dtype=self._voxtral_dtype,
                max_new_tokens=self._voxtral_max_new_tokens,
            )

        if backend == "voxtral_mlx":
            STATUS_LOG.info("Using MLX Voxtral transcriber on Apple Silicon.")
            from .voices.transcribe.voxtralmlx import VoxtralMlxTranscriber  # pylint: disable=import-outside-toplevel

            return VoxtralMlxTranscriber(
                model_size_or_path=self._voxtral_model_size,
                aligner_model_size_or_path=self._qwen_aligner_model_size,
                device=self._device,
                max_new_tokens=self._voxtral_max_new_tokens,
            )

        if sys.platform == "darwin":
            # If this is an Apple Silicon device, use the MLX Whisper transcriber
            if platform.processor() == "arm":
                try:
                    from .voices.transcribe.whispermlx import WhisperMlxTranscriber  # pylint: disable=import-outside-toplevel
                except ImportError:
                    STATUS_LOG.warning("WhisperMLX is not installed; falling back to faster-whisper on Apple Silicon.")
                else:
                    STATUS_LOG.info("Using WhisperMLX transcriber on Apple Silicon")
                    return WhisperMlxTranscriber(model_size_or_path=self._whisper_model_size)
            else:
                raise RuntimeError("Intel macOS is no longer supported; Apple Silicon is required on macOS.")

        STATUS_LOG.info("Using 'faster-whisper' transcriber.")
        from .voices.transcribe.faster_whisper import FasterWhisperTranscriber  # pylint: disable=import-outside-toplevel

        return FasterWhisperTranscriber(model_size_or_path=self._whisper_model_size, device=self._device)

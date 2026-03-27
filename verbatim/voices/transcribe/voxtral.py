import logging
import os
from time import perf_counter
from typing import Any, List, Optional, Tuple, cast

from numpy.typing import NDArray

from verbatim_audio.audio import samples_to_seconds

from ...transcript.words import Word
from .alignment_utils import project_timestamps_onto_transcript
from .qwen_asr import QWEN_LANGUAGE_NAMES
from .transcribe import Transcriber

LOG = logging.getLogger(__name__)


class VoxtralTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_size_or_path: str,
        aligner_model_size_or_path: str,
        device: str,
        dtype: str = "auto",
        max_new_tokens: int = 256,
    ):
        if device not in ("cpu", "cuda", "mps"):
            raise RuntimeError("Voxtral backend currently supports only 'cpu', 'cuda', and 'mps' devices.")

        try:
            import torch  # pylint: disable=import-outside-toplevel
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner  # pylint: disable=import-outside-toplevel
            from transformers import VoxtralForConditionalGeneration, VoxtralProcessor  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "Voxtral backend requires optional dependencies. Install `transformers`, `mistral-common`, `qwen-asr`, and `torch` "
                "to use transcriber_backend='voxtral'."
            ) from exc

        voxtral_dtype = self._resolve_dtype(torch_module=torch, dtype=dtype, device=device)
        if device == "cuda":
            device_map = "cuda:0"
        elif device == "mps":
            device_map = "mps"
        else:
            device_map = "cpu"

        offline_env = os.getenv("VERBATIM_OFFLINE", "0").lower() in ("1", "true", "yes")

        self._model_size_or_path = model_size_or_path
        self._processor: Any = VoxtralProcessor.from_pretrained(model_size_or_path, local_files_only=offline_env)  # nosec B615
        self._model: Any = VoxtralForConditionalGeneration.from_pretrained(  # nosec B615
            model_size_or_path,
            dtype=voxtral_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            local_files_only=offline_env,
        )
        self._aligner: Any = Qwen3ForcedAligner.from_pretrained(
            aligner_model_size_or_path,
            dtype=voxtral_dtype,
            device_map=device_map,
            local_files_only=offline_env,
        )
        self._device_map = device_map
        self._max_new_tokens = max_new_tokens
        self._warned_guess_language = False

    @staticmethod
    def _resolve_dtype(*, torch_module: Any, dtype: str, device: str) -> Any:
        if dtype == "auto":
            if device == "cuda":
                return torch_module.bfloat16
            if device == "mps" and hasattr(torch_module, "float16"):
                return torch_module.float16
            return torch_module.float32
        if not hasattr(torch_module, dtype):
            raise RuntimeError(f"Unsupported voxtral dtype: {dtype}")
        return getattr(torch_module, dtype)

    @staticmethod
    def _language_name(lang: str) -> Optional[str]:
        return QWEN_LANGUAGE_NAMES.get(lang)

    @staticmethod
    def _summarize_words(words: List[Word], max_text: int = 120) -> str:
        if not words:
            return "[]"
        text = "".join(word.word for word in words).replace("\n", " ")
        if len(text) > max_text:
            text = text[: max_text - 3] + "..."
        return f"[{words[0].start_ts}-{words[-1].end_ts}] n={len(words)} '{text}'"

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        _ = audio
        if len(lang) == 0:
            return "en", 1.0
        if len(lang) == 1:
            return lang[0], 1.0
        if not self._warned_guess_language:
            LOG.warning(
                "Voxtral backend does not provide native language identification. Use language_identifier_backend='mms' for reliable code-switching."
            )
            self._warned_guess_language = True
        return lang[0], 0.0

    def transcribe(
        self,
        *,
        audio: NDArray,
        lang: str,
        prompt: str,
        prefix: str,
        window_ts: int,
        audio_ts: int,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Optional[List[float]] = None,
    ) -> List[Word]:
        _ = whisper_beam_size, whisper_best_of, whisper_patience, whisper_temperatures

        mapped_language = self._language_name(lang)
        if mapped_language is None:
            raise RuntimeError(f"Language '{lang}' is not supported by the shared Qwen aligner mapping.")

        if prefix.strip() or prompt.strip():
            LOG.debug("Voxtral transcription request ignores prompt/prefix context for now: prompt=%r prefix=%r", prompt, prefix)

        transcribe_start = perf_counter()
        inputs: Any = self._processor.apply_transcription_request(
            language=mapped_language,
            audio=audio,
            model_id=self._model_size_or_path,
            sampling_rate=16000,
            format=["WAV"],
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device_map)
        generated_ids = self._model.generate(**cast(dict[str, Any], inputs), max_new_tokens=self._max_new_tokens)
        prompt_length = inputs["input_ids"].shape[1]
        transcript_ids = generated_ids[:, prompt_length:]
        transcript_text = self._processor.batch_decode(transcript_ids, skip_special_tokens=True)[0].strip()
        transcribe_elapsed_ms = (perf_counter() - transcribe_start) * 1000.0

        LOG.info(
            "Processed audio with duration %.3fs at window %.3fs-%.3fs (%d-%d samples) in %.1fms",
            max(0.0, len(audio) / 16000.0),
            samples_to_seconds(window_ts),
            samples_to_seconds(audio_ts),
            window_ts,
            audio_ts,
            transcribe_elapsed_ms,
        )

        if not transcript_text:
            return []

        align_start = perf_counter()
        alignment = self._aligner.align(
            audio=(audio, 16000),
            text=transcript_text,
            language=mapped_language,
        )[0]

        aligned_units: List[Tuple[int, int, str, float]] = []
        min_word_duration_samples = 800
        last_valid_end_ts = window_ts
        for item in alignment.items:
            start_ts = int(float(item.start_time) * 16000) + window_ts
            end_ts = int(float(item.end_time) * 16000) + window_ts
            if end_ts <= start_ts:
                end_ts = start_ts + min_word_duration_samples
            if start_ts < last_valid_end_ts:
                start_ts = last_valid_end_ts
                end_ts = max(end_ts, start_ts + min_word_duration_samples)
            if end_ts > audio_ts:
                continue
            aligned_units.append((start_ts, end_ts, item.text, 1.0))
            last_valid_end_ts = end_ts

        transcript_words = project_timestamps_onto_transcript(
            transcript_text=transcript_text,
            aligned_units=aligned_units,
            lang=lang,
            window_ts=window_ts,
            audio_ts=audio_ts,
        )
        align_elapsed_ms = (perf_counter() - align_start) * 1000.0

        LOG.debug(
            "Voxtral aligned words: lang=%s window=%d-%d words=%s",
            lang,
            window_ts,
            audio_ts,
            self._summarize_words(transcript_words),
        )
        LOG.info(
            "Window progress at %.3fs: words=%d transcribe_ms=%.1f align_ms=%.1f lang=%s",
            samples_to_seconds(audio_ts),
            len(transcript_words),
            transcribe_elapsed_ms,
            align_elapsed_ms,
            lang,
        )
        return transcript_words

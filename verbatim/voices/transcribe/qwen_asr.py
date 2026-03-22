import logging
from typing import Any, Dict, List, Optional, Tuple

from numpy.typing import NDArray

from ...transcript.words import Word
from .transcribe import Transcriber

LOG = logging.getLogger(__name__)

QWEN_LANGUAGE_NAMES: Dict[str, str] = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}

QWEN_LANGUAGE_CODES: Dict[str, str] = {name.lower(): code for code, name in QWEN_LANGUAGE_NAMES.items()}


class QwenAsrTranscriber(Transcriber):
    """Adapter that keeps all Qwen3-ASR specifics contained behind the transcriber interface."""

    def __init__(
        self,
        *,
        model_size_or_path: str,
        aligner_model_size_or_path: str,
        device: str,
        dtype: str = "auto",
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 256,
    ):
        if device not in ("cpu", "cuda"):
            raise RuntimeError("Qwen3-ASR backend currently supports only 'cpu' and 'cuda' devices.")

        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR backend requires optional dependencies. Install `qwen-asr` and `torch` to use transcriber_backend='qwen'."
            ) from exc

        qwen_dtype = self._resolve_dtype(torch_module=torch, dtype=dtype, device=device)
        device_map = "cuda:0" if device == "cuda" else "cpu"

        self._model = Qwen3ASRModel.from_pretrained(
            model_size_or_path,
            dtype=qwen_dtype,
            device_map=device_map,
            forced_aligner=aligner_model_size_or_path,
            forced_aligner_kwargs={
                "dtype": qwen_dtype,
                "device_map": device_map,
            },
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def _resolve_dtype(*, torch_module: Any, dtype: str, device: str) -> Any:
        if dtype == "auto":
            return torch_module.bfloat16 if device == "cuda" else torch_module.float32
        if not hasattr(torch_module, dtype):
            raise RuntimeError(f"Unsupported qwen dtype: {dtype}")
        return getattr(torch_module, dtype)

    @staticmethod
    def _language_name(lang: str) -> Optional[str]:
        return QWEN_LANGUAGE_NAMES.get(lang)

    @staticmethod
    def _language_code(language: Any, allowed_langs: List[str]) -> str:
        if isinstance(language, str):
            lowered = language.strip().lower()
            if lowered in QWEN_LANGUAGE_CODES:
                mapped = QWEN_LANGUAGE_CODES[lowered]
                if not allowed_langs or mapped in allowed_langs:
                    return mapped
            if language in allowed_langs:
                return language
        return allowed_langs[0] if allowed_langs else "en"

    @staticmethod
    def _get_field(item: Any, field_name: str) -> Any:
        if hasattr(item, field_name):
            return getattr(item, field_name)
        if isinstance(item, dict):
            return item.get(field_name)
        return None

    @staticmethod
    def _coerce_timestamp_text(transcript_text: str, cursor: int, unit_text: str) -> Tuple[str, int]:
        if cursor >= len(transcript_text):
            return unit_text, cursor

        if transcript_text.startswith(unit_text, cursor):
            end = cursor + len(unit_text)
            return transcript_text[cursor:end], end

        stripped_text = unit_text.strip()
        if stripped_text:
            found_at = transcript_text.find(stripped_text, cursor)
            if found_at != -1:
                end = found_at + len(stripped_text)
                return transcript_text[cursor:end], end

        fallback_end = min(len(transcript_text), cursor + len(unit_text))
        return transcript_text[cursor:fallback_end], fallback_end

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        if len(lang) == 0:
            return "en", 1.0
        if len(lang) == 1:
            return lang[0], 1.0

        results = self._model.transcribe(
            audio=(audio, 16000),
            language=None,
            return_time_stamps=False,
        )
        if not results:
            return lang[0], 0.0

        guessed_lang = self._language_code(self._get_field(results[0], "language"), lang)
        probability = 1.0 if guessed_lang in lang else 0.0
        return guessed_lang, probability

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

        qwen_language = self._language_name(lang)
        if qwen_language is None:
            raise RuntimeError(f"Language '{lang}' is not supported by the Qwen3-ASR adapter mapping.")

        context = prefix if prefix.strip() else prompt
        results = self._model.transcribe(
            audio=(audio, 16000),
            language=qwen_language,
            context=context,
            return_time_stamps=True,
        )
        if not results:
            return []

        result = results[0]
        transcript_text = self._get_field(result, "text") or ""
        time_stamps = self._get_field(result, "time_stamps")
        if not time_stamps:
            raise RuntimeError(
                f"Qwen3-ASR did not return timestamps for language '{lang}'. The current pipeline requires aligned timestamps."
            )

        transcript_words: List[Word] = []
        cursor = 0
        min_word_duration_samples = 800
        last_valid_end_ts = window_ts

        for time_stamp in time_stamps:
            unit_text = self._get_field(time_stamp, "text")
            start_time = self._get_field(time_stamp, "start_time")
            end_time = self._get_field(time_stamp, "end_time")
            if not isinstance(unit_text, str) or not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
                continue

            start_ts = int(float(start_time) * 16000) + window_ts
            end_ts = int(float(end_time) * 16000) + window_ts
            if end_ts <= start_ts:
                end_ts = start_ts + min_word_duration_samples
            if start_ts < last_valid_end_ts:
                start_ts = last_valid_end_ts
                end_ts = max(end_ts, start_ts + min_word_duration_samples)
            if end_ts > audio_ts:
                continue

            normalized_text, cursor = self._coerce_timestamp_text(transcript_text=transcript_text, cursor=cursor, unit_text=unit_text)
            if normalized_text == "":
                continue

            transcript_words.append(
                Word(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    word=normalized_text,
                    probability=1.0,
                    lang=lang,
                )
            )
            last_valid_end_ts = end_ts

        if transcript_words:
            joined_text = "".join(word.word for word in transcript_words)
            if transcript_text and joined_text != transcript_text:
                LOG.debug("Qwen timestamp units did not fully reconstruct transcript text: %r != %r", joined_text, transcript_text)

        return transcript_words

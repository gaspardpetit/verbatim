import logging
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from numpy.typing import NDArray

from verbatim_audio.audio import samples_to_seconds

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
            import torch  # pylint: disable=import-outside-toplevel
            from qwen_asr import Qwen3ASRModel  # pylint: disable=import-outside-toplevel
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
        self._configure_generation_padding()

    def _configure_generation_padding(self) -> None:
        """Avoid repeated Transformers warnings by ensuring pad_token_id is set."""
        model = getattr(self._model, "model", None)
        if model is None:
            return

        generation_config = getattr(model, "generation_config", None)
        model_config = getattr(model, "config", None)

        eos_token_id = None
        if generation_config is not None:
            eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None and model_config is not None:
            eos_token_id = getattr(model_config, "eos_token_id", None)
        if isinstance(eos_token_id, list) and eos_token_id:
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            return

        if generation_config is not None and getattr(generation_config, "pad_token_id", None) is None:
            generation_config.pad_token_id = eos_token_id
        if model_config is not None and getattr(model_config, "pad_token_id", None) is None:
            model_config.pad_token_id = eos_token_id

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
    def _split_transcript_text(transcript_text: str) -> List[str]:
        tokens: List[str] = []
        cursor = 0
        length = len(transcript_text)

        while cursor < length:
            token_start = cursor
            if transcript_text[cursor].isspace():
                while cursor < length and transcript_text[cursor].isspace():
                    cursor += 1
            while cursor < length and not transcript_text[cursor].isspace():
                cursor += 1
            token = transcript_text[token_start:cursor]
            if token:
                tokens.append(token)

        return tokens

    @staticmethod
    def _normalize_for_alignment(text: str) -> str:
        return "".join(char for char in text if char.isalnum())

    @classmethod
    def _find_matching_span(cls, transcript_text: str, cursor: int, normalized_text: str) -> Optional[Tuple[int, int]]:
        if not normalized_text:
            return None

        start: Optional[int] = None
        norm_index = 0
        first_char = normalized_text[0].lower()

        for index in range(cursor, len(transcript_text)):
            char = transcript_text[index]
            char_norm = char.lower() if char.isalnum() else ""

            if start is None:
                if char_norm == first_char:
                    start = index
                    norm_index = 1
                    if norm_index == len(normalized_text):
                        return start, index + 1
                continue

            if not char_norm:
                continue

            if norm_index < len(normalized_text) and char_norm == normalized_text[norm_index].lower():
                norm_index += 1
                if norm_index == len(normalized_text):
                    return start, index + 1
                continue

            if char_norm == first_char:
                start = index
                norm_index = 1
                if norm_index == len(normalized_text):
                    return start, index + 1
            else:
                start = None
                norm_index = 0

        return None

    @staticmethod
    def _split_leading_nonword(chunk: str) -> Tuple[str, str]:
        match = re.search(r"\w", chunk, re.UNICODE)
        if match is None:
            return chunk, ""
        if match.start() == 0:
            return "", chunk
        return chunk[: match.start()], chunk[match.start() :]

    @classmethod
    def _project_timestamps_onto_transcript(
        cls,
        *,
        transcript_text: str,
        aligned_units: List[Tuple[int, int, str, float]],
        lang: str,
        window_ts: int,
        audio_ts: int,
    ) -> List[Word]:
        if not transcript_text:
            return []

        words: List[Word] = []
        last_end_ts = window_ts
        cursor = 0

        for unit_start_ts, unit_end_ts, unit_text, unit_probability in aligned_units:
            normalized_unit = cls._normalize_for_alignment(unit_text)
            if not normalized_unit:
                continue
            if unit_end_ts > audio_ts:
                continue

            matching_span = cls._find_matching_span(transcript_text=transcript_text, cursor=cursor, normalized_text=normalized_unit)
            if matching_span is None:
                continue

            _match_start, match_end = matching_span
            raw_chunk = transcript_text[cursor:match_end]
            leading_nonword, lexical_chunk = cls._split_leading_nonword(raw_chunk)

            if leading_nonword and words:
                words[-1].word += leading_nonword
                words[-1].end_ts = max(words[-1].end_ts, unit_start_ts)
            elif leading_nonword:
                lexical_chunk = leading_nonword + lexical_chunk

            if not lexical_chunk:
                cursor = match_end
                continue

            start_ts = unit_start_ts
            end_ts = unit_end_ts
            probability = unit_probability
            if start_ts < last_end_ts:
                start_ts = last_end_ts
                end_ts = max(end_ts, start_ts)

            words.append(
                Word(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    word=lexical_chunk,
                    probability=probability,
                    lang=lang,
                )
            )
            last_end_ts = end_ts
            cursor = match_end

        if cursor < len(transcript_text) and words:
            tail = transcript_text[cursor:]
            words[-1].word += tail

        return words

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

        process_start = perf_counter()
        audio_duration_seconds = max(0.0, len(audio) / 16000.0)
        context = prefix if prefix.strip() else prompt
        results = self._model.transcribe(
            audio=(audio, 16000),
            language=qwen_language,
            context=context,
            return_time_stamps=True,
        )
        elapsed_ms = (perf_counter() - process_start) * 1000.0
        LOG.info(
            "Processed audio with duration %.3fs at window %.3fs-%.3fs (%d-%d samples) in %.1fms",
            audio_duration_seconds,
            samples_to_seconds(window_ts),
            samples_to_seconds(audio_ts),
            window_ts,
            audio_ts,
            elapsed_ms,
        )
        if not results:
            return []

        result = results[0]
        transcript_text = self._get_field(result, "text") or ""
        time_stamps = self._get_field(result, "time_stamps")
        if not time_stamps:
            raise RuntimeError(f"Qwen3-ASR did not return timestamps for language '{lang}'. The current pipeline requires aligned timestamps.")

        aligned_units: List[Tuple[int, int, str, float]] = []
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

            aligned_units.append((start_ts, end_ts, unit_text, 1.0))
            last_valid_end_ts = end_ts

        transcript_words = self._project_timestamps_onto_transcript(
            transcript_text=transcript_text,
            aligned_units=aligned_units,
            lang=lang,
            window_ts=window_ts,
            audio_ts=audio_ts,
        )
        if transcript_words:
            joined_text = "".join(word.word for word in transcript_words)
            if transcript_text and joined_text != transcript_text:
                LOG.debug("Qwen timestamp units did not fully reconstruct transcript text: %r != %r", joined_text, transcript_text)

        words_count = len(transcript_words)
        elapsed_seconds = max(elapsed_ms / 1000.0, 1e-9)
        words_per_second = words_count / elapsed_seconds
        LOG.info(
            "Window progress at %.3fs: words=%d throughput=%.2f words/s lang=%s",
            samples_to_seconds(audio_ts),
            words_count,
            words_per_second,
            lang,
        )

        return transcript_words

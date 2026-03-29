import logging
import sys
from time import perf_counter
from typing import Any, List, Optional, Tuple

from numpy.typing import NDArray

from verbatim_audio.audio import samples_to_seconds

from ...transcript.words import Word
from .alignment_utils import normalize_for_alignment, project_timestamps_onto_transcript, split_transcript_text
from .transcribe import Transcriber

LOG = logging.getLogger(__name__)

if sys.platform == "darwin":
    from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor
else:
    VoxtralForConditionalGeneration = None
    VoxtralProcessor = None


class VoxtralMlxTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_size_or_path: str,
        aligner_model_size_or_path: str,
        device: str,
        max_new_tokens: int = 256,
    ):
        if sys.platform != "darwin":
            raise RuntimeError("Voxtral MLX backend is only supported on macOS (MLX).")
        if VoxtralForConditionalGeneration is None or VoxtralProcessor is None:
            raise RuntimeError("Voxtral MLX backend requires the optional `mlx-voxtral` package on macOS.")
        _ = aligner_model_size_or_path, device

        self._model_repo = self._resolve_model_repo(model_size_or_path)
        self._model: Any = VoxtralForConditionalGeneration.from_pretrained(self._model_repo)
        self._processor: Any = VoxtralProcessor.from_pretrained(self._model_repo)
        self._max_new_tokens = max_new_tokens
        self._warned_guess_language = False

    @staticmethod
    def _resolve_model_repo(model_size_or_path: str) -> str:
        return model_size_or_path

    @staticmethod
    def _summarize_words(words: List[Word], max_text: int = 120) -> str:
        if not words:
            return "[]"
        text = "".join(word.word for word in words).replace("\n", " ")
        if len(text) > max_text:
            text = text[: max_text - 3] + "..."
        return f"[{words[0].start_ts}-{words[-1].end_ts}] n={len(words)} '{text}'"

    @staticmethod
    def _approximate_aligned_units(*, transcript_text: str, window_ts: int, audio_ts: int) -> List[Tuple[int, int, str, float]]:
        tokens = split_transcript_text(transcript_text)
        lexical_tokens = [token for token in tokens if normalize_for_alignment(token)]
        if not lexical_tokens:
            return []

        total_duration = max(1, audio_ts - window_ts)
        total_weight = sum(max(1, len(normalize_for_alignment(token))) for token in lexical_tokens)
        remaining_duration = total_duration
        remaining_weight = total_weight
        current_ts = window_ts
        aligned_units: List[Tuple[int, int, str, float]] = []

        for index, token in enumerate(lexical_tokens):
            token_weight = max(1, len(normalize_for_alignment(token)))
            if index == len(lexical_tokens) - 1:
                next_ts = audio_ts
            else:
                token_duration = max(1, int(round(remaining_duration * token_weight / max(1, remaining_weight))))
                next_ts = min(audio_ts, current_ts + token_duration)
            if next_ts <= current_ts:
                next_ts = min(audio_ts, current_ts + 1)
            aligned_units.append((current_ts, next_ts, token, 1.0))
            remaining_duration = max(0, audio_ts - next_ts)
            remaining_weight = max(0, remaining_weight - token_weight)
            current_ts = next_ts

        if aligned_units and aligned_units[-1][1] < audio_ts:
            last_start, _last_end, last_text, last_prob = aligned_units[-1]
            aligned_units[-1] = (last_start, audio_ts, last_text, last_prob)
        return aligned_units

    def _eos_token_ids(self) -> List[int]:
        eos_token_ids = getattr(self._processor.tokenizer, "eos_token_ids", [2, 4, 32000])
        if isinstance(eos_token_ids, int):
            return [eos_token_ids]
        return list(eos_token_ids)

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        _ = audio
        if len(lang) == 0:
            return "en", 1.0
        if len(lang) == 1:
            return lang[0], 1.0
        if not self._warned_guess_language:
            LOG.warning(
                "Voxtral MLX backend does not provide trusted native language identification. "
                "Use language_identifier_backend='mms' for reliable code-switching."
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

        if prefix.strip() or prompt.strip():
            LOG.debug("Voxtral MLX transcription ignores prompt/prefix context for now: prompt=%r prefix=%r", prompt, prefix)

        transcribe_start = perf_counter()
        inputs = self._processor.apply_transcrition_request(
            audio=audio,
            language=lang,
            sampling_rate=16000,
        )
        outputs = self._model.generate(
            input_ids=inputs.input_ids,
            input_features=inputs.input_features,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        input_len = inputs.input_ids.shape[1]
        transcript_text = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
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
        aligned_units = self._approximate_aligned_units(
            transcript_text=transcript_text,
            window_ts=window_ts,
            audio_ts=audio_ts,
        )
        transcript_words = project_timestamps_onto_transcript(
            transcript_text=transcript_text,
            aligned_units=aligned_units,
            lang=lang,
            window_ts=window_ts,
            audio_ts=audio_ts,
        )
        align_elapsed_ms = (perf_counter() - align_start) * 1000.0

        LOG.debug(
            "Voxtral MLX aligned words: lang=%s window=%d-%d words=%s",
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

import logging
import sys
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, cast

from numpy.typing import NDArray

from ...audio.audio import samples_to_seconds
from ...transcript.words import Word
from .transcribe import Transcriber

LOG = logging.getLogger(__name__)

if sys.platform == "darwin":
    # MLX / mlx-whisper only available on macOS
    import mlx.core as MX_CORE
    from mlx_whisper.audio import (
        N_FRAMES,
        N_SAMPLES,
        SAMPLE_RATE,
    )
    from mlx_whisper.audio import (
        log_mel_spectrogram as LOG_MEL_SPECTROGRAM,
    )
    from mlx_whisper.audio import (
        pad_or_trim as PAD_OR_TRIM,
    )
    from mlx_whisper.transcribe import ModelHolder as MODEL_HOLDER
    from mlx_whisper.transcribe import transcribe as MLX_TRANSCRIBE
else:
    MX_CORE = None
    MODEL_HOLDER = None
    MLX_TRANSCRIBE = None
    LOG_MEL_SPECTROGRAM = None
    PAD_OR_TRIM = None
    N_FRAMES = None
    N_SAMPLES = None
    SAMPLE_RATE = None


TranscribeResult = Dict[str, Any]
TranscribeSegment = Dict[str, Any]
TranscribeWord = Dict[str, Any]


class WhisperMlxTranscriber(Transcriber):
    """
    MLX-based Whisper backend.

    Notes:
    - Beam search is NOT implemented in mlx-whisper yet, so we deliberately
      ignore beam_size/best_of/patience here to avoid runtime errors.
    - Those knobs are still exposed on the class for API compatibility with
      FasterWhisperTranscriber, but effectively no-ops on this backend.
    """

    model_path: str
    whisper_beam_size: int
    whisper_best_of: int
    whisper_patience: float
    whisper_temperatures: List[float]

    def __init__(
        self,
        *,
        model_size_or_path: str,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Optional[List[float]] = None,
    ):
        if sys.platform != "darwin":
            raise RuntimeError("WhisperMlxTranscriber is only supported on macOS (MLX).")

        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.model_path = f"mlx-community/whisper-{model_size_or_path}-mlx"
        self.whisper_beam_size = whisper_beam_size
        self.whisper_best_of = whisper_best_of
        self.whisper_patience = whisper_patience
        self.whisper_temperatures = whisper_temperatures

    # -------------------------------------------------------------------------
    # Language detection
    # -------------------------------------------------------------------------

    def _get_mlx_model(self):
        """
        Use mlx-whisper's ModelHolder cache to avoid re-loading the model.
        """
        if MODEL_HOLDER is None or MX_CORE is None:
            raise RuntimeError("mlx-whisper is not available on this platform.")
        dtype = MX_CORE.float16
        return MODEL_HOLDER.get_model(self.model_path, dtype=dtype)

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        """
        Use the model's own detect_language() on the first ~30s of audio,
        similar to how mlx_whisper.transcribe does it internally.
        """
        if audio is None or audio.size == 0:
            fallback = lang[0] if lang else "en"
            LOG.warning("Empty audio in guess_language; falling back to %s", fallback)
            return fallback, 0.0

        model = self._get_mlx_model()

        if N_SAMPLES is None or N_FRAMES is None:
            raise RuntimeError("mlx-whisper spectrogram constants are not available on this platform.")

        # Non-multilingual models: force "en", but respect allowed list.
        if not getattr(model, "is_multilingual", True):
            chosen = "en"
            if lang and chosen not in lang:
                chosen = lang[0]
            LOG.info("Non-multilingual MLX model, using language=%s", chosen)
            return chosen, 1.0

        # Build log-mel spectrogram and run detect_language
        if LOG_MEL_SPECTROGRAM is None or PAD_OR_TRIM is None or MX_CORE is None:
            raise RuntimeError("mlx-whisper audio helpers are not available on this platform.")

        mel = LOG_MEL_SPECTROGRAM(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        mel_segment = PAD_OR_TRIM(mel, N_FRAMES, axis=-2).astype(MX_CORE.float16)
        detect_result = model.detect_language(mel_segment)
        if not isinstance(detect_result, tuple) or len(detect_result) != 2:
            raise RuntimeError("Unexpected detect_language return format")

        probs_raw = detect_result[1]
        if not isinstance(probs_raw, Mapping):
            raise RuntimeError("Language probabilities are not a mapping")

        probs: Dict[str, float] = {}
        for code, prob in probs_raw.items():
            if isinstance(code, str):
                try:
                    probs[code] = float(prob)
                except (TypeError, ValueError):
                    continue

        if not probs:
            raise RuntimeError("Language probabilities mapping is empty")

        best_lang = None
        best_prob = 0.0
        if lang:
            for code, p in probs.items():
                if code in lang and p > best_prob:
                    best_lang = code
                    best_prob = float(p)

        if best_lang is None:
            # No restriction or no allowed language found: take global max
            top_entry = max(probs.items(), key=lambda item: item[1])
            best_lang, best_prob = top_entry[0], float(top_entry[1])

        if best_lang is None:
            raise RuntimeError("Language detection failed to produce a candidate")

        LOG.info("Detected language '%s' with probability %.3f", best_lang, best_prob)
        return best_lang, best_prob

    # -------------------------------------------------------------------------
    # Transcription
    # -------------------------------------------------------------------------

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
        LOG.info(
            "Transcribing audio window: window_ts=%d, audio_ts=%d (lang=%s)",
            window_ts,
            audio_ts,
            lang,
        )

        # Robust empty check
        if audio is None or audio.size == 0:
            LOG.warning("Empty or invalid audio chunk received")
            return []

        # Temperatures: sequence of temperatures for fallback sampling
        if whisper_temperatures is None:
            temperature_source = self.whisper_temperatures
        else:
            temperature_source = whisper_temperatures
        temperatures: Tuple[float, ...] = tuple(temperature_source)

        # Keep these for logging / API consistency, but we do NOT pass them
        # to mlx_transcribe because beam search is not implemented there.
        beam_size = whisper_beam_size or self.whisper_beam_size
        best_of = whisper_best_of or self.whisper_best_of
        patience = whisper_patience or self.whisper_patience
        LOG.debug(
            "MLX backend ignoring beam search params: beam_size=%d, best_of=%d, patience=%f",
            beam_size,
            best_of,
            patience,
        )

        verbose = LOG.getEffectiveLevel() <= logging.INFO

        # Call mlx-whisper transcribe WITHOUT beam-related options.
        # This uses greedy / sampling decoding only.
        if MLX_TRANSCRIBE is None or SAMPLE_RATE is None:
            raise RuntimeError("mlx-whisper transcribe function is not available on this platform.")

        raw_result = MLX_TRANSCRIBE(
            audio,
            task="transcribe",
            path_or_hf_repo=self.model_path,
            language=lang,
            initial_prompt=prompt or None,
            word_timestamps=True,
            verbose=(True if verbose else None),
            temperature=temperatures,
            # Default-ish thresholds; keep explicit for transparency.
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",
            hallucination_silence_threshold=None,
        )

        result: TranscribeResult = cast(TranscribeResult, raw_result)
        segments_obj = result.get("segments")
        if not isinstance(segments_obj, list):
            LOG.warning("MLX transcription returned no segments list")
            return []
        if not segments_obj:
            LOG.warning("MLX transcription returned empty segments")
            return []

        segments: List[TranscribeSegment] = [segment for segment in segments_obj if isinstance(segment, dict)]
        if not segments:
            LOG.warning("MLX transcription segments are not dicts")
            return []

        transcript_words: List[Word] = []
        current_segment_lang = lang  # MLX does not expose per-segment language

        last_valid_end_ts = window_ts
        min_word_duration_ms = 50
        min_word_duration_samples = int(SAMPLE_RATE * min_word_duration_ms / 1000.0)

        for segment in segments:
            lang_value = segment.get("language", current_segment_lang)
            segment_lang = lang_value if isinstance(lang_value, str) else current_segment_lang
            if segment_lang != current_segment_lang:
                LOG.info("Language switch detected: %s -> %s", current_segment_lang, segment_lang)
                current_segment_lang = segment_lang

            words_obj = segment.get("words", [])
            if not isinstance(words_obj, list):
                continue

            word_items: List[TranscribeWord] = [word for word in words_obj if isinstance(word, dict)]

            for word_data in word_items:
                start_val = word_data.get("start")
                end_val = word_data.get("end")
                word_text = word_data.get("word")
                if not isinstance(start_val, (int, float)) or not isinstance(end_val, (int, float)) or not isinstance(word_text, str):
                    continue

                # MLX word timestamps are in seconds -> convert to samples
                raw_start_ts = int(start_val * SAMPLE_RATE) + window_ts
                raw_end_ts = int(end_val * SAMPLE_RATE) + window_ts

                start_ts = raw_start_ts
                end_ts = raw_end_ts

                # Repair invalid timestamps (end <= start)
                if end_ts <= start_ts:
                    LOG.info(
                        "Fixing invalid timestamps for word %r: start=%d end=%d",
                        word_text,
                        start_ts,
                        end_ts,
                    )

                    if start_ts > last_valid_end_ts:
                        # Valid start, just missing/invalid end → minimal duration
                        end_ts = start_ts + min_word_duration_samples
                    else:
                        # Shift to after last valid word and estimate duration
                        start_ts = last_valid_end_ts
                        word_len = len(word_text.strip())
                        est_dur = max(
                            min_word_duration_samples,
                            (word_len * min_word_duration_samples) // 2,
                        )
                        end_ts = start_ts + est_dur

                    LOG.info(
                        "Fixed timestamps for word %r: start=%d end=%d",
                        word_text,
                        start_ts,
                        end_ts,
                    )

                # Enforce the known "valid audio" range
                if end_ts > audio_ts:
                    LOG.debug(
                        "Skipping word %r: end_ts=%d > audio_ts=%d",
                        word_text,
                        end_ts,
                        audio_ts,
                    )
                    continue

                last_valid_end_ts = end_ts

                probability_val = word_data.get("probability", 1.0)
                if not isinstance(probability_val, (int, float)):
                    probability_val = 1.0

                word = Word(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    word=word_text,
                    probability=float(probability_val),
                    lang=current_segment_lang,
                )

                LOG.debug(
                    "Word %r: end_ts=%d (%.3fs)",
                    word.word,
                    word.end_ts,
                    samples_to_seconds(word.end_ts),
                )
                transcript_words.append(word)

        return transcript_words

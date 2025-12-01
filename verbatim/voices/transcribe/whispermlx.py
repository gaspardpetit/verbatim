import logging
import sys
from typing import List, Optional, Tuple, Union

from numpy.typing import NDArray

from ...audio.audio import samples_to_seconds
from ...transcript.words import Word
from .transcribe import Transcriber

LOG = logging.getLogger(__name__)

if sys.platform == "darwin":
    # MLX / mlx-whisper only available on macOS
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder, transcribe as mlx_transcribe
    from mlx_whisper.audio import (
        N_FRAMES,
        N_SAMPLES,
        SAMPLE_RATE,
        log_mel_spectrogram,
        pad_or_trim,
    )
else:
    mx = None
    ModelHolder = None
    mlx_transcribe = None


class WhisperMlxTranscriber(Transcriber):
    """
    MLX-based Whisper backend.

    Notes:
    - Beam search is NOT implemented in mlx-whisper yet, so we deliberately
      ignore beam_size/best_of/patience here to avoid runtime errors.
    - Those knobs are still exposed on the class for API compatibility with
      FasterWhisperTranscriber, but effectively no-ops on this backend.
    """

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
        if ModelHolder is None:
            raise RuntimeError("mlx-whisper is not available on this platform.")
        dtype = mx.float16
        return ModelHolder.get_model(self.model_path, dtype=dtype)

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

        # Non-multilingual models: force "en", but respect allowed list.
        if not getattr(model, "is_multilingual", True):
            chosen = "en"
            if lang and chosen not in lang:
                chosen = lang[0]
            LOG.info("Non-multilingual MLX model, using language=%s", chosen)
            return chosen, 1.0

        # Build log-mel spectrogram and run detect_language
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(mel_segment)  # dict: lang_code -> prob

        best_lang = None
        best_prob = 0.0
        if lang:
            for code, p in probs.items():
                if code in lang and p > best_prob:
                    best_lang = code
                    best_prob = float(p)

        if best_lang is None:
            # No restriction or no allowed language found: take global max
            best_lang = max(probs, key=probs.get)
            best_prob = float(probs[best_lang])

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
            whisper_temperatures = self.whisper_temperatures
        if isinstance(whisper_temperatures, list):
            temperatures: Union[float, tuple] = tuple(whisper_temperatures)
        else:
            temperatures = whisper_temperatures

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
        result = mlx_transcribe(
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

        if not result or "segments" not in result:
            LOG.warning("MLX transcription returned no segments")
            return []

        transcript_words: List[Word] = []
        current_segment_lang = lang  # MLX does not expose per-segment language

        last_valid_end_ts = window_ts
        min_word_duration_ms = 50
        min_word_duration_samples = int(SAMPLE_RATE * min_word_duration_ms / 1000.0)

        for segment in result["segments"]:
            segment_lang = segment.get("language", current_segment_lang)
            if segment_lang != current_segment_lang:
                LOG.info("Language switch detected: %s -> %s", current_segment_lang, segment_lang)
                current_segment_lang = segment_lang

            for word_data in segment.get("words", []):
                # MLX word timestamps are in seconds -> convert to samples
                raw_start_ts = int(word_data["start"] * SAMPLE_RATE) + window_ts
                raw_end_ts = int(word_data["end"] * SAMPLE_RATE) + window_ts

                start_ts = raw_start_ts
                end_ts = raw_end_ts

                # Repair invalid timestamps (end <= start)
                if end_ts <= start_ts:
                    LOG.info(
                        "Fixing invalid timestamps for word %r: start=%d end=%d",
                        word_data["word"],
                        start_ts,
                        end_ts,
                    )

                    if start_ts > last_valid_end_ts:
                        # Valid start, just missing/invalid end → minimal duration
                        end_ts = start_ts + min_word_duration_samples
                    else:
                        # Shift to after last valid word and estimate duration
                        start_ts = last_valid_end_ts
                        word_len = len(word_data["word"].strip())
                        est_dur = max(
                            min_word_duration_samples,
                            (word_len * min_word_duration_samples) // 2,
                        )
                        end_ts = start_ts + est_dur

                    LOG.info(
                        "Fixed timestamps for word %r: start=%d end=%d",
                        word_data["word"],
                        start_ts,
                        end_ts,
                    )

                # Enforce the known "valid audio" range
                if end_ts > audio_ts:
                    LOG.debug(
                        "Skipping word %r: end_ts=%d > audio_ts=%d",
                        word_data["word"],
                        end_ts,
                        audio_ts,
                    )
                    continue

                last_valid_end_ts = end_ts

                word = Word(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    word=word_data["word"],
                    probability=float(word_data.get("probability", 1.0)),
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

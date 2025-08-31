import logging
import os
from typing import Dict, List, Optional, Tuple

import whisper
from numpy.typing import NDArray
from whisper.model import Whisper

from ...transcript.words import Word
from .transcribe import Transcriber, WhisperConfig

LOG = logging.getLogger(__name__)


class WhisperTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_size_or_path: str,
        device: str,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Optional[List[float]] = None,
    ):
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.device = device
        self.whisper_beam_size = whisper_beam_size
        self.whisper_best_of = whisper_best_of
        self.whisper_patience = whisper_patience
        self.whisper_temperatures = whisper_temperatures
        # Honor offline and cache directory if provided
        offline_env = os.getenv("VERBATIM_OFFLINE", "0").lower() in ("1", "true", "yes")
        resolved_model: str = model_size_or_path
        if offline_env and not os.path.exists(model_size_or_path):
            # Try to resolve from cache directory
            whisper_cache = os.getenv(
                "WHISPER_CACHE_DIR",
                os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "whisper"),
            )
            candidate = os.path.join(whisper_cache, f"{model_size_or_path}.pt")
            if os.path.exists(candidate):
                resolved_model = candidate
            else:
                raise RuntimeError(
                    f"Offline mode is enabled and Whisper model '{model_size_or_path}' is not present in cache: {candidate}"
                )

        self.model: Whisper = whisper.load_model(resolved_model, device=device)

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        padded_audio = whisper.pad_or_trim(audio)
        mel_spectrogram = whisper.log_mel_spectrogram(padded_audio, n_mels=self.model.dims.n_mels).to(self.model.device)

        lang_probs: Dict[str, float]
        _, lang_probs = self.model.detect_language(mel=mel_spectrogram)  # pyright: ignore[reportAssignmentType]
        candidates: List[Tuple[str, float]] = [(k, v) for (k, v) in lang_probs.items() if k in lang]
        best_lang = max(candidates, key=lambda x: x[1], default=None)
        if best_lang is None:
            return "en", 0.0

        return best_lang[0], best_lang[1]

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
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        use_fp16 = self.device == "cuda"
        verbose: Optional[bool] = None
        if LOG.getEffectiveLevel() < logging.INFO:
            verbose = True
        elif LOG.getEffectiveLevel() < logging.WARN:
            verbose = False  # will still display a progress bar
        else:
            verbose = None

        whisper_config: WhisperConfig = WhisperConfig()

        options = whisper.DecodingOptions(
            task=whisper_config.task,
            language=lang,
            temperature=tuple(whisper_temperatures),  # pyright: ignore[reportArgumentType]
            sample_len=None,
            best_of=whisper_best_of,
            beam_size=whisper_beam_size,
            patience=whisper_patience,
            length_penalty=whisper_config.length_penalty,
            prompt=prompt,
            prefix=prefix,
            suppress_tokens=whisper_config.suppress_tokens,
            suppress_blank=whisper_config.suppress_blank,
            without_timestamps=False,
            max_initial_timestamp=1.0,
            fp16=use_fp16,
        )

        transcript = self.model.transcribe(
            audio=audio,
            word_timestamps=True,
            verbose=verbose,
            compression_ratio_threshold=whisper_config.compression_ratio_threshold,
            logprob_threshold=whisper_config.logprob_threshold,
            no_speech_threshold=whisper_config.no_speech_threshold,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            prepend_punctuations=whisper_config.prepend_punctuations,
            append_punctuations=whisper_config.append_punctuations,
            clip_timestamps=[0.0],
            hallucination_silence_threshold=None,
            **options.__dict__,
        )
        words: List[Word] = []
        segment: Dict
        for segment in transcript["segments"]:  # pyright: ignore[reportAssignmentType]
            # read optional fields for completeness, but ignore values
            _ = segment.get("id")  # noqa: F841  pyright: ignore[reportAssignmentType]
            _ = segment.get("seek")  # noqa: F841  pyright: ignore[reportAssignmentType]
            _ = segment.get("start")  # noqa: F841  pyright: ignore[reportAssignmentType]
            _ = segment.get("end")  # noqa: F841  pyright: ignore[reportAssignmentType]
            _ = segment.get("text")  # noqa: F841  pyright: ignore[reportAssignmentType]
            _ = segment.get("temperature")  # noqa: F841  pyright: ignore[reportAssignmentType]
            _ = segment.get("avg_logprob")  # noqa: F841  pyright: ignore[reportAssignmentType]
            _ = segment.get("compression_ratio")  # noqa: F841  pyright: ignore[reportAssignmentType]
            _ = segment.get("no_speech_prob")  # noqa: F841  pyright: ignore[reportAssignmentType]
            segment_words: List[Dict] = segment.get("words")  # pyright: ignore[reportAssignmentType]
            for word in segment_words:
                word_start: float = word.get("start")  # pyright: ignore[reportAssignmentType]
                word_end: float = word.get("end")  # pyright: ignore[reportAssignmentType]
                word_text: str = word.get("word")  # pyright: ignore[reportAssignmentType]
                word_probability: float = word.get("probability")  # pyright: ignore[reportAssignmentType]

                start_ts = int(word_start * 16000) + window_ts
                end_ts = int(word_end * 16000) + window_ts
                words.append(
                    Word(
                        start_ts=start_ts,
                        end_ts=end_ts,
                        word=word_text,
                        probability=word_probability,
                        lang=lang,
                    )
                )

        return words

# PS08_verbatim/verbatim/voices/transcribe/whispermlx.py

import logging
import sys
from typing import List, Tuple, Optional

from numpy.typing import NDArray

from ...audio.audio import samples_to_seconds
from ...transcript.words import Word
from .transcribe import Transcriber
from verbatim.config import Config

if sys.platform == "darwin":
    # pylint: disable=import-error
    from mlx_whisper import transcribe
else:
    transcribe = None  # pylint: disable=invalid-name

LOG = logging.getLogger(__name__)


class WhisperMlxTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_size_or_path: str,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Optional[List[float]] = None,
    ):
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.model_path = f"mlx-community/whisper-{model_size_or_path}-mlx"
        self.whisper_beam_size = whisper_beam_size
        self.whisper_best_of = whisper_best_of
        self.whisper_patience = whisper_patience
        self.whisper_temperatures = whisper_temperatures

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        result = transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            language=None,  # Trigger language detection
            verbose=None,  # pyright:  ignore[reportOptionalCall]
            task="transcribe",
            no_speech_threshold=0.6,
        )

        detected = result["language"]
        # Check if detected language is in allowed languages
        if detected in lang:
            LOG.info(f"Detected language: {detected}")
            return detected, 1.0

        # If not in allowed languages, use first allowed language
        LOG.warning(f"Detected language {detected} not in allowed languages {lang}, using {lang[0]}")
        LOG.info(f"Detected language: {detected}")
        prob = 1.0 if samples_to_seconds(len(audio)) > 8.0 else 0.1
        return lang[0], prob

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
        LOG.info(f"Transcribing audio window: window_ts={window_ts}, audio_ts={audio_ts}")

        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        temperatures = tuple(whisper_temperatures) if isinstance(whisper_temperatures, list) else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        show_progress = LOG.getEffectiveLevel() <= logging.INFO

        LOG.info(f"Transcribing with temperatures: {temperatures}")
        LOG.info(f"Transcribing with prefix: {prefix}")

        result = transcribe(
            audio,
            task="transcribe",
            path_or_hf_repo=self.model_path,
            language=lang,
            initial_prompt=prompt if prompt else None,
            prefix=prefix if prefix else None,
            word_timestamps=True,
            verbose=(True if show_progress else None),
            temperature=temperatures,
            no_speech_threshold=0.6,
        )

        transcript_words: List[Word] = []
        current_segment_lang = lang
        last_end = window_ts

        # Process segments and words
        for segment in result["segments"]:
            segment_lang = segment.get("language", lang)
            if segment_lang != current_segment_lang:
                LOG.info(f"Language switch detected: {current_segment_lang} -> {segment_lang}")
                current_segment_lang = segment_lang

            segment_words = []
            for word_data in segment.get("words", []):
                start_ts = int(word_data["start"] * Config.sampling_rate) + window_ts
                end_ts = int(word_data["end"] * Config.sampling_rate) + window_ts

                # Adjust timestamps if invalid
                if end_ts <= start_ts:
                    start_ts = last_end
                    end_ts = start_ts + max(1600, len(word_data["word"].strip()) * 100)

                # Only include word if timestamps are valid and within range
                if start_ts >= window_ts and end_ts <= audio_ts and end_ts > start_ts:
                    word = Word(
                        start_ts=start_ts,
                        end_ts=end_ts,
                        word=word_data["word"],
                        probability=word_data.get("probability", 1.0),
                        lang=current_segment_lang,
                    )
                    transcript_words.append(word)
                    last_end = end_ts

        return transcript_words

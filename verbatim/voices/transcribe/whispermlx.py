# PS08_verbatim/verbatim/voices/transcribe/whispermlx.py

import logging
import sys
from typing import List, Tuple, Union

import numpy as np

from ...audio.audio import samples_to_seconds
from ...transcript.words import Word
from .transcribe import Transcriber

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
        whisper_temperatures: Union[None, List[float]] = None,
    ):
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.model_path = f"mlx-community/whisper-{model_size_or_path}-mlx"
        self.whisper_beam_size = whisper_beam_size
        self.whisper_best_of = whisper_best_of
        self.whisper_patience = whisper_patience
        self.whisper_temperatures = whisper_temperatures

    def guess_language(self, audio: np.ndarray, lang: List[str]) -> Tuple[str, float]:
        result = transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            language=None,  # Trigger language detection
            verbose=None,
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
        audio: np.ndarray,
        lang: str,
        prompt: str,
        prefix: str,
        window_ts: int,
        audio_ts: int,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Union[None, List[float]] = None,
    ) -> List[Word]:
        LOG.info(f"Transcribing audio window: window_ts={window_ts}, audio_ts={audio_ts}")

        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # When whisper_temperatures is of type list
        if isinstance(whisper_temperatures, list):
            # Transform into tuple of floats
            temperatures = tuple(whisper_temperatures)
        else:
            temperatures = whisper_temperatures

        # Set up transcription options
        show_progress = LOG.getEffectiveLevel() <= logging.INFO

        result = transcribe(
            audio,
            task="transcribe",
            path_or_hf_repo=self.model_path,
            language=lang,
            initial_prompt=prompt if prompt else None,
            word_timestamps=True,
            # Not yet implemented in MLX Whisper, see https://github.com/ml-explore/mlx-examples/issues/846
            # beam_size=whisper_beam_size,
            # patience=whisper_patience, # requires beam_size
            best_of=whisper_best_of,
            verbose=(True if show_progress else None),  # None = don't even show progress bar
            temperature=temperatures,
            no_speech_threshold=0.6,
        )

        # Convert results to Word objects
        transcript_words: List[Word] = []
        current_segment_lang = lang

        # Process segments and words
        for segment in result["segments"]:
            # Check if segment has a different language
            segment_lang = segment.get("language", lang)
            if segment_lang != current_segment_lang:
                LOG.info(f"Language switch detected: {current_segment_lang} -> {segment_lang}")
                current_segment_lang = segment_lang

            for word_data in segment.get("words", []):
                # Create Word object with correct language tag and timestamp offset
                start_ts = int(word_data["start"] * 16000) + window_ts
                end_ts = int(word_data["end"] * 16000) + window_ts

                # Validate timestamps
                if end_ts <= start_ts:
                    LOG.warning(f"Invalid timestamps for word '{word_data['word']}': start={start_ts}, end={end_ts}")
                    continue

                if end_ts > audio_ts:
                    LOG.debug(f"Skipping word '{word_data['word']}' as it ends after audio_ts")
                    continue

                # Create Word object with timestamp offset
                word = Word(
                    start_ts=int(word_data["start"] * 16000) + window_ts,
                    end_ts=int(word_data["end"] * 16000) + window_ts,
                    word=word_data["word"],
                    probability=word_data.get("probability", 1.0),
                    lang=current_segment_lang,
                )

                # Log word timetamp comparison
                LOG.info(f"Word '{word.word}': end_ts={word.end_ts}, audio_ts={audio_ts}")
                transcript_words.append(word)

        return transcript_words

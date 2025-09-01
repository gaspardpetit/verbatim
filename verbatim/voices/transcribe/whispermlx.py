# PS08_verbatim/verbatim/voices/transcribe/whispermlx.py

import logging
import sys
from typing import List, Optional, Tuple

from numpy.typing import NDArray

from verbatim.audio.audio import samples_to_seconds, seconds_to_samples
from verbatim.audio.settings import AUDIO_PARAMS
from verbatim.transcript.words import Word

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

        # Add check for empty or invalid audio
        if audio.size == 0 or audio is None:
            LOG.warning("Empty or invalid audio chunk received")
            return []

        # Handle temperatures
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if isinstance(whisper_temperatures, list):
            # Transform into tuple of floats
            temperatures = tuple(whisper_temperatures)
        else:
            temperatures = whisper_temperatures

        # Call MLX Whisper
        result = transcribe(
            audio,
            task="transcribe",
            path_or_hf_repo=self.model_path,
            language=lang,
            initial_prompt=prompt if prompt else None,
            word_timestamps=True,
            # Not yet implemented, see https://github.com/ml-explore/mlx-examples/issues/846
            # beam_size=whisper_beam_size,
            # patience=whisper_patience, # requires beam_size
            # best_of=whisper_best_of, # leads to many std::bad_cast errors
            verbose=(True if LOG.getEffectiveLevel() <= logging.INFO else None),  # pyright: ignore[reportOptionalCall]
            temperature=temperatures,
            # Below are the default values for transparency, see
            # https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/transcribe.py#L62
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",
            hallucination_silence_threshold=None,
        )

        if not result or "segments" not in result:
            LOG.warning("Transcription returned no valid results, no segments found")
            return []

        # Convert results to Word objects
        transcript_words: List[Word] = []
        current_segment_lang = lang

        # Keep track of last valid timestamp to help fix invalid ones
        last_valid_end_ts = window_ts
        min_word_duration = 50  # Minimum duration for a word in milliseconds
        min_word_duration_samples = int(min_word_duration * AUDIO_PARAMS.sample_rate / 1000)

        # Process segments and words
        for segment in result["segments"]:
            # Check if segment has a different language
            segment_lang = segment.get("language", lang)
            if segment_lang != current_segment_lang:
                LOG.info(f"Language switch detected: {current_segment_lang} -> {segment_lang}")
                current_segment_lang = segment_lang

            for word_data in segment.get("words", []):
                # Create Word object with correct language tag and timestamp offset
                raw_start_ts = seconds_to_samples(word_data["start"]) + window_ts
                raw_end_ts = seconds_to_samples(word_data["end"]) + window_ts

                # Fix invalid timestamps
                if raw_end_ts <= raw_start_ts:
                    LOG.info(f"Fixing invalid timestamps for word '{word_data['word']}': start={raw_start_ts}, end={raw_end_ts}")

                    # If we have a valid start time but invalid end time
                    if raw_start_ts > last_valid_end_ts:
                        # Assign a reasonable minimum duration
                        raw_end_ts = raw_start_ts + min_word_duration_samples
                    # If both timestamps are invalid or start <= last_end
                    else:
                        # Start from the last valid end time
                        raw_start_ts = last_valid_end_ts
                        # Estimate duration based on word length (approximation)
                        word_length = len(word_data["word"].strip())
                        word_duration = max(min_word_duration_samples, word_length * min_word_duration_samples // 2)
                        raw_end_ts = raw_start_ts + word_duration

                    LOG.info(f"Fixed timestamps for word '{word_data['word']}': start={raw_start_ts}, end={raw_end_ts}")

                if raw_end_ts > audio_ts:
                    LOG.debug(f"Skipping word '{word_data['word']}' as it ends after audio_ts")
                    continue

                # Create Word object with timestamp offset
                word = Word(
                    start_ts=seconds_to_samples(word_data["start"]) + window_ts,
                    end_ts=seconds_to_samples(word_data["end"]) + window_ts,
                    word=word_data["word"],
                    probability=word_data.get("probability", 1.0),
                    lang=current_segment_lang,
                )

                # Log word timetamp comparison
                LOG.debug(f"Word '{word.word}': end_ts={word.end_ts}, audio_ts={audio_ts}")
                transcript_words.append(word)

        # Return collected words
        return transcript_words

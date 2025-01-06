import logging
from typing import List, Tuple

import numpy as np
from faster_whisper import WhisperModel
from ...audio.audio import samples_to_seconds
from ...transcript.words import VerbatimWord
from .transcribe import Transcriber, WhisperConfig

LOG = logging.getLogger(__name__)


class FasterWhisperTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_size_or_path: str,
        device: str,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: List[float] = None,
    ):
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        if device == "cpu":
            self.whisper_model = WhisperModel(
                model_size_or_path=model_size_or_path,
                device=device,
                compute_type="default",
                cpu_threads=16,
            )
        else:
            self.whisper_model = WhisperModel(
                model_size_or_path=model_size_or_path,
                device=device,
                compute_type="float16",
            )

        self.whisper_beam_size = whisper_beam_size
        self.whisper_best_of = whisper_best_of
        self.whisper_patience = whisper_patience
        self.whisper_temperatures = whisper_temperatures

    def guess_language(self, audio: np.array, lang: List[str]) -> Tuple[str, float]:
        language, language_probability, all_language_probs = (
            self.whisper_model.detect_language(audio=audio)
        )
        if language in lang:
            LOG.info(f"detected '{language}' with probability {language_probability}")
            return language, language_probability

        guess_lang = lang[0]
        guess_prob = 0
        for lang_iter in lang:
            for t in all_language_probs:
                if t[0] == lang_iter:
                    if t[1] > guess_prob:
                        guess_lang = t[0]
                        guess_prob = t[1]
                        LOG.info(
                            f"detected '{lang_iter}' with probability {guess_prob}"
                        )
        return guess_lang, guess_prob

    def transcribe(
        self,
        *,
        audio: np.array,
        lang: str,
        prompt: str,
        prefix: str,
        window_ts: int,
        audio_ts: int,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: List[float] = None,
    ) -> List[VerbatimWord]:
        LOG.info(f"Transcription Prefix: {prefix}")
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        show_progress = LOG.getEffectiveLevel() <= logging.INFO
        whisper_config: WhisperConfig = WhisperConfig()
        segment_iter, info = self.whisper_model.transcribe(
            audio=audio,
            language=lang,
            task=whisper_config.task,
            log_progress=show_progress,
            beam_size=whisper_beam_size,
            best_of=whisper_best_of,
            patience=whisper_patience,
            length_penalty=whisper_config.length_penalty,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            temperature=whisper_temperatures,
            compression_ratio_threshold=whisper_config.compression_ratio_threshold,
            log_prob_threshold=whisper_config.logprob_threshold,
            no_speech_threshold=whisper_config.no_speech_threshold,
            condition_on_previous_text=False,
            prompt_reset_on_temperature=0.0,
            initial_prompt=prompt,
            prefix=prefix,
            suppress_blank=whisper_config.suppress_blank,
            suppress_tokens=whisper_config.suppress_tokens,
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=True,
            prepend_punctuations=whisper_config.prepend_punctuations,
            append_punctuations=whisper_config.append_punctuations,
            multilingual=False,
            vad_filter=False,
            vad_parameters=None,
            max_new_tokens=None,
            chunk_length=None,
            clip_timestamps="0.0",
            hallucination_silence_threshold=None,
            hotwords=None,
            language_detection_threshold=0.5,
            language_detection_segments=1,
        )

        LOG.debug(f"info={info}")

        transcript_words: List[VerbatimWord] = []
        for segment in segment_iter:
            for w in segment.words:
                word = VerbatimWord.from_word(word=w, lang=lang, ts_offset=window_ts)
                if word.end_ts > audio_ts:
                    continue

                LOG.debug(
                    f"[{word.start_ts} ({samples_to_seconds(word.start_ts)})]: {w.word}"
                )
                transcript_words.append(word)

        return transcript_words

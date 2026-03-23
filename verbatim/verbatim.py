# pylint: disable=too-many-lines
import logging
import math
import os
import sys
import traceback
import wave
from dataclasses import dataclass, field
from io import StringIO
from time import perf_counter
from typing import Any, Generator, List, Optional, TextIO, Tuple

import numpy as np
from colorama import Fore
from numpy.typing import NDArray

from verbatim.logging_utils import get_status_logger
from verbatim_audio.audio import samples_to_seconds
from verbatim_audio.sources.audiosource import AudioSource, AudioStream
from verbatim_files.format.factory import configure_writers
from verbatim_files.format.json import read_utterances
from verbatim_files.format.txt import (
    COLORSCHEME_ACKNOWLEDGED,
    COLORSCHEME_NONE,
    COLORSCHEME_UNACKNOWLEDGED,
    COLORSCHEME_UNCONFIRMED,
    TranscriptFormatter,
)
from verbatim_files.format.writer import (
    LanguageStyle,
    ProbabilityStyle,
    SpeakerStyle,
    TimestampStyle,
    TranscriptWriterConfig,
)
from verbatim_transcript import (
    LanguageDetectionRequest,
    LanguageDetectionResult,
    TranscriberProtocol,
    TranscriptionWindowResult,
    VadFn,
    detect_language,
)

from .config import Config
from .language_id import create_language_identifier
from .models import Models
from .status_types import StatusHook, StatusProgress, StatusUpdate
from .transcript.idprovider import CounterIdProvider, IdProvider
from .transcript.sentences import BoundedSentenceTokenizer, SentenceTokenizer
from .transcript.words import Utterance, Word

# pylint: disable=unused-import
from .voices.transcribe.transcribe import APPEND_PUNCTUATIONS, PREPEND_PUNCTUATIONS

# Optional type-only import to avoid pyannote dependency at runtime
Annotation = Any  # pylint: disable=invalid-name

# Configure logger
LOG = logging.getLogger(__name__)
STATUS_LOG = get_status_logger()


@dataclass
class WhisperHistory:
    size: int = 12
    transcript_history: List[List[Word]] = field(default_factory=list)

    @staticmethod
    def _words_summary(words: List[Word], max_text: int = 60) -> str:
        if len(words) == 0:
            return "[]"
        text = "".join(w.word for w in words).replace(os.linesep, " ")
        if len(text) > max_text:
            text = text[: max_text - 3] + "..."
        return f"[{words[0].start_ts}-{words[-1].end_ts}] n={len(words)} '{text}'"

    @staticmethod
    def _words_text(words: List[Word]) -> str:
        return "".join(w.word for w in words).replace(os.linesep, " ")

    def add(self, transcription: List[Word]):
        self.transcript_history.append(transcription)
        if len(self.transcript_history) > self.size:
            self.transcript_history.pop(0)

    @staticmethod
    def advance_transcript(timestamp: int, transcript: List[Word]) -> List[Word]:
        return [w for w in transcript if w.end_ts >= timestamp]

    def advance(self, timestamp: int):
        previous_history = self.transcript_history
        advanced_history = [self.advance_transcript(timestamp, transcript) for transcript in previous_history]
        self.transcript_history = [h for h in advanced_history if len(h) > 0]
        if LOG.isEnabledFor(logging.DEBUG):
            changes = []
            for before, after in zip(previous_history, advanced_history):
                if before != after:
                    changes.append(f"{self._words_summary(before)} -> {self._words_summary(after)}")
            if changes:
                LOG.debug("Advanced transcript history at %d: %s", timestamp, " | ".join(changes))

    @staticmethod
    def confirm_transcript(
        current_words: List[Word],
        transcript: List[Word],
        prefix: List[Word],
        after_ts: int,
    ) -> List[Word]:
        c_index = 0
        while c_index < len(current_words):
            if (c_index < len(prefix) and prefix[c_index].word == current_words[c_index].word) or (current_words[c_index].start_ts <= after_ts):
                c_index = c_index + 1
            else:
                break

        p_index = 0
        while p_index < len(transcript):
            if (p_index < len(prefix) and prefix[p_index].word == transcript[p_index].word) or (transcript[p_index].start_ts <= after_ts):
                p_index = p_index + 1
            else:
                break

        confirmed_words = []

        while p_index < len(transcript) and c_index < len(current_words):
            if transcript[p_index].word.strip().lower() == current_words[c_index].word.strip().lower():
                confirmed_words += [current_words[c_index]]
                p_index += 1
                c_index += 1
            else:
                break
        if len(confirmed_words) > 0:
            LOG.debug(
                "Confirmed candidate: matched=%d transcript=%s current=%s confirmed=%s",
                len(confirmed_words),
                WhisperHistory._words_summary(transcript),
                WhisperHistory._words_summary(current_words),
                WhisperHistory._words_summary(confirmed_words),
            )
        return confirmed_words

    def confirm(
        self,
        current_words: List[Word],
        prefix: List[Word],
        after_ts: int,
    ) -> List[Word]:
        return max(
            (
                self.confirm_transcript(
                    current_words=current_words,
                    transcript=transcript,
                    prefix=prefix,
                    after_ts=after_ts,
                )
                for transcript in self.transcript_history
            ),
            key=len,
            default=[],
        )


class RollingWindow:
    array: NDArray

    def __init__(self, window_size, dtype=np.float32):
        self.reset(window_size=window_size, dtype=dtype)

    def reset(self, window_size=-1, dtype=None):
        if window_size == -1:
            window_size = len(self.array)
        if dtype is None:
            dtype = self.array.dtype
        self.array = np.zeros(window_size, dtype=dtype)


@dataclass
class State:
    confirmed_ts: int
    acknowledged_ts: int
    window_ts: int
    audio_ts: int
    processing_started_at: float
    processing_started_window_ts: int
    transcript_candidate_history: WhisperHistory
    rolling_window: RollingWindow
    acknowledged_words: List[Word]
    unconfirmed_words: List[Word]
    acknowledged_utterances: List[Utterance]
    unacknowledged_utterances: List[Utterance]
    last_transcribe_lang: str
    skip_silences: bool = True
    speaker_embeddings: Optional[List] = None
    working_prefix_no_ext: str = "out"
    timing_totals_ms: dict[str, float] = field(init=False)
    last_transcribe_metrics_ms: dict[str, float] = field(init=False)

    def __init__(self, config: Config, working_prefix_no_ext: str = "out"):
        self.config = config
        self.confirmed_ts = -1
        self.acknowledged_ts = -1
        self.window_ts = 0  # Window timestamp in samples
        self.audio_ts = 0  # Global timestamp in samples
        self.processing_started_at = perf_counter()
        self.processing_started_window_ts = 0
        self.transcript_candidate_history = WhisperHistory()
        self.acknowledged_words = []
        self.unconfirmed_words = []
        self.acknowledged_utterances = []
        self.unacknowledged_utterances = []
        self.last_transcribe_lang = ""
        self.speaker_embeddings = []
        self.working_prefix_no_ext = working_prefix_no_ext
        self.utterance_id: IdProvider = CounterIdProvider(prefix="utt")
        self.timing_totals_ms = {
            "detect": 0.0,
            "transcribe": 0.0,
            "confirm": 0.0,
            "sentence": 0.0,
            "acknowledge": 0.0,
            "speaker": 0.0,
            "output": 0.0,
            "advance": 0.0,
            "total": 0.0,
        }
        self.last_transcribe_metrics_ms = {
            "detect_ms": 0.0,
            "transcribe_ms": 0.0,
            "confirm_ms": 0.0,
        }

        window_size = self.config.sampling_rate * self.config.window_duration  # Total samples in 30 seconds
        self.rolling_window = RollingWindow(window_size=window_size, dtype=np.float32)  # Initialize empty rolling window

    def advance_audio_window(self, offset: int):
        if offset <= 0:
            return
        # Do not advance beyond available audio; keep window_ts <= audio_ts
        max_advance = max(0, self.audio_ts - self.window_ts)
        if offset > max_advance:
            LOG.warning(
                f"advance_audio_window requested {samples_to_seconds(offset)}s but only {samples_to_seconds(max_advance)}s available; clamping."
            )
            offset = max_advance
        if offset == 0:
            return
        LOG.debug(
            f"Shifting rolling window by {offset} samples ({samples_to_seconds(offset)}s) "
            f"to offset {self.window_ts + offset} ({samples_to_seconds(self.window_ts + offset)}s); "
            f"Window now has {samples_to_seconds(self.audio_ts - self.window_ts)} seconds of audio."
        )

        self.rolling_window.array = np.roll(self.rolling_window.array, -offset)
        self.rolling_window.array[-offset:] = 0
        self.window_ts += offset

        LOG.debug("Window now has %s seconds of audio.", samples_to_seconds(self.audio_ts - self.window_ts))

    def append_audio_to_window(self, audio_chunk: NDArray):
        if audio_chunk.size == 0:
            LOG.warning("Received empty audio chunk, skipping.")
            return

        # Convert stereo to mono if necessary
        LOG.debug(f"Audio chunk shape before mono conversion: {audio_chunk.shape}")
        if len(audio_chunk.shape) > 1 and audio_chunk.shape[1] > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        LOG.debug(f"Audio chunk shape after mono conversion: {audio_chunk.shape}")

        chunk_size = len(audio_chunk)
        window_size = len(self.rolling_window.array)
        # Guard against a temporarily inconsistent state where window_ts > audio_ts
        insertion_start = max(0, self.audio_ts - self.window_ts)
        insertion_end = insertion_start + chunk_size
        if self.audio_ts + chunk_size <= self.window_ts + window_size:
            # Append to the current rolling window
            self.rolling_window.array[insertion_start:insertion_end] = audio_chunk
        else:
            # Shift the window and append new audio
            shift_amount = self.audio_ts + chunk_size - (self.window_ts + window_size)
            LOG.warning(f"failed to acknowledge transcription within audio buffer of {samples_to_seconds(window_size)} seconds")
            self.advance_audio_window(shift_amount)
            self.skip_silences = True
            self.rolling_window.array[-chunk_size:] = audio_chunk

        self.audio_ts += chunk_size


class Verbatim:
    state: State
    config: Config
    vad_callback: Optional[VadFn]
    status_hook: Optional[StatusHook]

    def __init__(
        self,
        config: Config,
        *,
        models=None,
        vad_callback: Optional[VadFn] = None,
        transcriber: Optional[TranscriberProtocol] = None,
        status_hook: Optional[StatusHook] = None,
    ):
        self.config = config
        self.state = State(config)
        self.status_hook = status_hook
        if models is None:
            models = Models(
                device=config.device,
                whisper_model_size=config.whisper_model_size,
                stream=config.stream,
                transcriber=transcriber,
                transcriber_backend=config.transcriber_backend,
                qwen_asr_model_size=config.qwen_asr_model_size,
                qwen_aligner_model_size=config.qwen_aligner_model_size,
                qwen_dtype=config.qwen_dtype,
                qwen_max_inference_batch_size=config.qwen_max_inference_batch_size,
                qwen_max_new_tokens=config.qwen_max_new_tokens,
            )
        self.models = models
        self.language_identifier = create_language_identifier(config=config, models=self.models)
        if vad_callback is not None:
            self.vad_callback = vad_callback
        elif hasattr(self.models, "vad"):
            # default Silero VAD path; keeps current behavior while enabling injection later
            self.vad_callback = self.models.vad.find_activity  # type: ignore[attr-defined]
        else:
            self.vad_callback = None

    def _emit_status(self, update: StatusUpdate) -> None:
        if self.status_hook is None:
            return
        try:
            self.status_hook(update)
        except Exception:  # pylint: disable=broad-exception-caught
            LOG.exception("Status hook failed")

    def _emit_utterance_status(
        self,
        *,
        utterance: Utterance,
        unacknowledged: List[Utterance],
        unconfirmed: List[Word],
    ) -> None:
        self._emit_status(
            StatusUpdate(
                state="transcribing",
                utterance=utterance,
                unacknowledged_utterances=unacknowledged,
                unconfirmed_words=unconfirmed,
            )
        )

    @staticmethod
    def _get_total_samples(audio_stream: AudioStream) -> Optional[int]:
        total_samples = getattr(audio_stream, "total_samples", None)
        if total_samples is None:
            return None
        start_offset = getattr(audio_stream, "start_offset", 0) or 0
        end_sample = getattr(audio_stream, "end_sample", None)
        if end_sample is not None:
            total_samples = min(total_samples, end_sample)
        return max(0, total_samples - start_offset)

    def _emit_transcribing_progress(self, audio_stream: AudioStream) -> None:
        total_samples = self._get_total_samples(audio_stream)
        if total_samples is None or total_samples <= 0:
            return
        start_offset = getattr(audio_stream, "start_offset", 0) or 0
        current_samples = max(0, self.state.window_ts - start_offset)
        self._emit_status(
            StatusUpdate(
                state="transcribing",
                progress=StatusProgress(
                    current=samples_to_seconds(current_samples),
                    finish=samples_to_seconds(total_samples),
                    units="seconds",
                ),
            )
        )

    def _detect_language_for_audio(self, *, audio: NDArray, window_ts: int, audio_ts: int, timestamp: int) -> LanguageDetectionResult:
        request = LanguageDetectionRequest(
            audio=audio,
            lang=self.config.lang,
            timestamp=timestamp,
            window_ts=window_ts,
            audio_ts=audio_ts,
        )
        return detect_language(request=request, guess_fn=self.language_identifier.guess_language)

    def _read_full_audio(self, audio_stream: AudioStream) -> NDArray:
        chunks: List[NDArray] = []
        chunk_length = max(1, self.config.window_duration)
        while audio_stream.has_more():
            chunk = audio_stream.next_chunk(chunk_length=chunk_length)
            if chunk.size == 0:
                continue
            if len(chunk.shape) > 1 and chunk.shape[1] > 1:
                chunk = np.mean(chunk, axis=1)
            chunks.append(chunk.astype(np.float32))
        if len(chunks) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _summarize_words(words: List[Word], max_text: int = 80) -> str:
        if len(words) == 0:
            return "[]"
        text = "".join(word.word for word in words).replace(os.linesep, " ")
        if len(text) > max_text:
            text = text[: max_text - 3] + "..."
        langs = sorted({word.lang for word in words if word.lang})
        langs_summary = f" langs={','.join(langs)}" if langs else ""
        return f"[{words[0].start_ts}-{words[-1].end_ts}] n={len(words)}{langs_summary} '{text}'"

    @staticmethod
    def _words_text(words: List[Word]) -> str:
        return "".join(word.word for word in words).replace(os.linesep, " ")

    @staticmethod
    def _summarize_utterance(utterance: Utterance, max_text: int = 80) -> str:
        text = utterance.text.replace(os.linesep, " ")
        if len(text) > max_text:
            text = text[: max_text - 3] + "..."
        return f"{utterance.utterance_id}[{utterance.start_ts}-{utterance.end_ts}] '{text}'"

    @classmethod
    def _summarize_utterances(cls, utterances: List[Utterance], max_items: int = 4) -> str:
        if len(utterances) == 0:
            return "[]"
        parts = [cls._summarize_utterance(utterance) for utterance in utterances[:max_items]]
        if len(utterances) > max_items:
            parts.append(f"...(+{len(utterances) - max_items})")
        return "[" + ", ".join(parts) + "]"

    def skip_leading_silence(self, max_skip: int, min_speech_duration_ms: int = 500) -> int:
        if self.vad_callback is None:
            LOG.debug("VAD callback not configured; skipping silence detection and keeping current window.")
            return self.state.window_ts

        min_speech_duration_ms = 750
        min_speech_duration_samples = 16000 * min_speech_duration_ms // 1000
        audio_samples = self.state.audio_ts - self.state.window_ts
        voice_segments = self.vad_callback(self.state.rolling_window.array[0:audio_samples], min_speech_duration_ms, 100)
        LOG.debug(f"Voice segments: {voice_segments}")
        if len(voice_segments) == 0:
            # preserve a bit of audio at the end that may have been too short just because it is truncated
            padding = min_speech_duration_samples
            remaining_audio = self.state.audio_ts - self.state.window_ts - padding
            advance_to = remaining_audio
            # pylint: disable=consider-using-min-builtin
            if max_skip < advance_to:
                advance_to = max_skip
            LOG.debug(
                f"Skipping silences between {samples_to_seconds(self.state.window_ts):.2f}"
                f" and {samples_to_seconds(self.state.window_ts + advance_to):.2f}"
            )

            self.state.advance_audio_window(advance_to)
            return self.state.audio_ts

        voice_start = voice_segments[0]["start"]
        voice_end = voice_segments[0]["end"]
        voice_length = voice_end - voice_start

        # skip leading silences
        if voice_start > 0:
            # pylint: disable=consider-using-min-builtin
            advance_to = voice_start
            if max_skip < advance_to:
                advance_to = max_skip
            LOG.debug(
                f"Skipping silences between {samples_to_seconds(self.state.window_ts):.2f}"
                f" and {samples_to_seconds(self.state.window_ts + advance_to):.2f}"
            )
            self.state.advance_audio_window(advance_to)
            return voice_length + self.state.window_ts

        return voice_length + self.state.window_ts

    def dump_window_to_file(self, filename: str = "debug_window.wav"):
        window = self.state.rolling_window.array
        LOG.debug("Dumping current window to file for debugging.")
        # pylint: disable=no-member
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.config.sampling_rate)
            wf.writeframes((window * 32768.0).astype(np.int16).tobytes())
        LOG.debug("Finished writing window to file.")

    @staticmethod
    def align_words_to_sentences(id_provider: IdProvider, sentences: list[str], window_words: list[Word]) -> list[Utterance]:
        # Concatenate all sentences into a single string
        full_text_sentences = "".join(sentences)
        # Concatenate all words from window_words
        full_text_words = "".join(w.word for w in window_words)

        # Basic validation step (optional but recommended)
        if full_text_sentences != full_text_words:
            raise ValueError("The joined text from sentences and window_words do not match.")

        # We'll iterate through sentences and pick words until we match them character-for-character.
        result = []
        current_char_index = 0
        current_word_index = 0
        word_char_offset = 0

        def remove_spaces_and_punctuation(string: str) -> str:
            return string.translate(str.maketrans("", "", " " + PREPEND_PUNCTUATIONS + APPEND_PUNCTUATIONS))

        # For each sentence, we want to collect a sublist of VerbatimWord
        for sentence_index, sentence in enumerate(sentences):
            raw_sentence = sentence
            # igonre punctuation and spaces
            sentence = remove_spaces_and_punctuation(sentence)
            sentence_length = len(sentence.strip())
            target_end = current_char_index + sentence_length

            sentence_words = []
            # Accumulate words until we have accounted for all chars of this sentence
            while current_char_index < target_end:
                if current_word_index >= len(window_words):
                    raise ValueError("Ran out of words while aligning sentences.")

                w = window_words[current_word_index]
                word_text = remove_spaces_and_punctuation(w.word)
                word_length = len(word_text)

                # Words that resolve to only punctuation/whitespace carry no characters; assign them directly.
                if word_length == 0:
                    sentence_words.append(w)
                    current_word_index += 1
                    word_char_offset = 0
                    continue

                remaining_word_chars = word_length - word_char_offset
                if remaining_word_chars <= 0:
                    # Safety: reset offset if we somehow consumed the full word previously.
                    word_char_offset = 0
                    continue

                remaining_sentence_chars = target_end - current_char_index
                if remaining_sentence_chars <= 0:
                    break

                if remaining_word_chars <= remaining_sentence_chars:
                    # Entire remaining portion of the word fits in this sentence.
                    current_char_index += remaining_word_chars
                    sentence_words.append(w)
                    current_word_index += 1
                    word_char_offset = 0
                else:
                    # Only part of the word belongs to this sentence; defer adding the word until it is fully consumed.
                    LOG.debug(
                        "Sentence alignment split inside word: sentence_index=%d raw=%r normalized=%r target_end=%d "
                        "word=%r word_range=%d-%d word_char_offset=%d remaining_word_chars=%d remaining_sentence_chars=%d",
                        sentence_index,
                        raw_sentence,
                        sentence,
                        target_end,
                        w.word,
                        w.start_ts,
                        w.end_ts,
                        word_char_offset,
                        remaining_word_chars,
                        remaining_sentence_chars,
                    )
                    current_char_index += remaining_sentence_chars
                    word_char_offset += remaining_sentence_chars
                    break

            # Now we have all the words for this sentence
            if len(sentence_words) > 0:
                utterance = Utterance.from_words(utterance_id=id_provider.next(), words=sentence_words)
                LOG.debug(
                    "Sentence alignment result: sentence_index=%d raw=%r normalized=%r utterance=%r words=%s",
                    sentence_index,
                    raw_sentence,
                    sentence,
                    utterance.text,
                    [word.word for word in sentence_words],
                )
                result.append(utterance)
            else:
                LOG.debug(
                    "Sentence alignment produced empty utterance: sentence_index=%d raw=%r normalized=%r target_end=%d",
                    sentence_index,
                    raw_sentence,
                    sentence,
                    target_end,
                )

        # At the end, `result` should be a List[List[VerbatimWord]]
        return result

    @staticmethod
    def words_to_sentences(word_tokenizer: SentenceTokenizer, window_words: List[Word], id_provider: IdProvider) -> list[Utterance]:
        sentences = []
        if len(window_words) == 0:
            return []

        for tok in word_tokenizer.split(words=window_words):
            sentences += [tok]

        LOG.debug(
            "Sentence tokenizer output: tokenizer=%s words=%s sentences=%s",
            type(word_tokenizer).__name__,
            [word.word for word in window_words],
            sentences,
        )

        utterances = Verbatim.align_words_to_sentences(id_provider=id_provider, sentences=sentences, window_words=window_words)
        return utterances

    def guess_language(self, timestamp: int) -> Tuple[str, float, int]:
        request = LanguageDetectionRequest(
            audio=self.state.rolling_window.array,
            lang=self.config.lang,
            timestamp=timestamp,
            window_ts=self.state.window_ts,
            audio_ts=self.state.audio_ts,
        )
        result: LanguageDetectionResult = detect_language(request=request, guess_fn=self.language_identifier.guess_language)
        return (result.language, result.probability, result.samples_used)

    def transcribe_window(self) -> Tuple[List[Word], List[Word]]:
        LOG.debug("Starting transcription of audio chunk.")
        LOG.debug("Window Start Time: %s", samples_to_seconds(self.state.window_ts))
        LOG.debug("Confirmed Time: %s", samples_to_seconds(self.state.confirmed_ts))
        LOG.debug("Acknowledged Time: %s", samples_to_seconds(self.state.acknowledged_ts))
        LOG.debug("Valid audio range: 0.0 - %s", samples_to_seconds(self.state.audio_ts - self.state.window_ts))

        acknowledged_words_in_window = WhisperHistory.advance_transcript(timestamp=self.state.window_ts, transcript=self.state.acknowledged_words)
        detect_start = perf_counter()
        lang, _prob, used_samples_for_language = self.guess_language(timestamp=max(0, self.state.acknowledged_ts))
        self.state.last_transcribe_lang = lang
        detect_ms = (perf_counter() - detect_start) * 1000.0
        prefix_text = ""
        for word in [w for u in self.state.unacknowledged_utterances for w in u.words]:
            if word.lang != lang:
                break
            prefix_text += word.word
        whisper_prompt = self.config.whisper_prompts[lang] if lang in self.config.whisper_prompts else self.config.whisper_prompts["en"]

        try:
            transcribe_start = perf_counter()
            transcript_words = self.models.transcriber.transcribe(
                audio=self.state.rolling_window.array[0 : self.state.audio_ts - self.state.window_ts],
                lang=lang,
                prompt=whisper_prompt,
                prefix=prefix_text,
                window_ts=self.state.window_ts,
                audio_ts=self.state.audio_ts,
                whisper_beam_size=self.config.whisper_beam_size,
                whisper_best_of=self.config.whisper_best_of,
                whisper_patience=self.config.whisper_patience,
                whisper_temperatures=self.config.whisper_temperatures,
            )
            transcribe_ms = (perf_counter() - transcribe_start) * 1000.0
        except RuntimeError as e:
            LOG.warning(f"Transcription failed with RuntimeError: {str(e)}. Skipping this chunk.")
            self.state.last_transcribe_metrics_ms = {
                "detect_ms": detect_ms,
                "transcribe_ms": 0.0,
                "confirm_ms": 0.0,
            }
            return [], []

        if len(transcript_words) > 0 and len(self.config.lang) > 1:
            # make sure that the first word detected is within the audio samples used
            # to guess the language; otherwise, language might have been guess from a short utterance such
            # as "hmm hmm" followed by a pause and then actual words

            if transcript_words[0].start_ts >= self.state.window_ts + used_samples_for_language:
                # So the words detected are not part of the samples evaluated when guessing the language
                # in this unfortunate situation, we could simply perform language detected on the range of
                # words detected, but there is an non-zero chance that the transcription skipped words
                # because we suggested the wrong langauge. So instead, we try the following:
                # 1. Transcribe in each of the supported language;
                # 2. Confirm with language detection that we used the appropriate language
                # 3. if language detection does not match, ignore the transcripton
                # 4. within the transcription that matched, process the one that starts with the earliest timestamp

                best_first_word_ts = self.state.audio_ts
                best_transcript = transcript_words
                best_lang = lang
                for test_lang in self.config.lang:
                    if test_lang == lang:
                        first_word_ts = transcript_words[0].start_ts
                        confirmed_lang, _prob, _used_samples_for_language = self.guess_language(timestamp=max(0, first_word_ts))
                        if confirmed_lang == test_lang:
                            if transcript_words[0].start_ts < best_first_word_ts:
                                best_first_word_ts = transcript_words[0].start_ts
                                best_transcript = transcript_words
                                best_lang = lang
                    else:
                        alt_prefix_text = ""
                        for word in [w for u in self.state.unacknowledged_utterances for w in u.words]:
                            if word.lang != test_lang:
                                break
                            alt_prefix_text += word.word
                        alt_whisper_prompt = (
                            self.config.whisper_prompts[test_lang] if test_lang in self.config.whisper_prompts else self.config.whisper_prompts["en"]
                        )
                        alt_transcribe_start = perf_counter()
                        alt_transcript_words = self.models.transcriber.transcribe(
                            audio=self.state.rolling_window.array[0 : self.state.audio_ts - self.state.window_ts],
                            lang=test_lang,
                            prompt=alt_whisper_prompt,
                            prefix=alt_prefix_text,
                            window_ts=self.state.window_ts,
                            audio_ts=self.state.audio_ts,
                            whisper_beam_size=self.config.whisper_beam_size,
                            whisper_best_of=self.config.whisper_best_of,
                            whisper_patience=self.config.whisper_patience,
                            whisper_temperatures=self.config.whisper_temperatures,
                        )
                        transcribe_ms += (perf_counter() - alt_transcribe_start) * 1000.0
                        if len(alt_transcript_words) > 0:
                            first_word_ts = alt_transcript_words[0].start_ts
                            confirmed_lang, _prob, _used_samples_for_language = self.guess_language(timestamp=max(0, first_word_ts))
                            if confirmed_lang == test_lang:
                                if alt_transcript_words[0].start_ts < best_first_word_ts:
                                    best_first_word_ts = alt_transcript_words[0].start_ts
                                    best_transcript = alt_transcript_words
                                    best_lang = test_lang
                lang = best_lang
                transcript_words = best_transcript

        confirm_start = perf_counter()
        self.state.transcript_candidate_history.advance(self.state.window_ts)
        confirmed_words = self.state.transcript_candidate_history.confirm(
            current_words=transcript_words,
            after_ts=self.state.acknowledged_ts - 1,
            prefix=acknowledged_words_in_window,
        )
        self.state.transcript_candidate_history.add(transcription=transcript_words)

        if len(confirmed_words) > 0:
            self.state.confirmed_ts = confirmed_words[-1].start_ts
            LOG.debug(f"Confirmed ts: {self.state.confirmed_ts} ({samples_to_seconds(self.state.confirmed_ts)})")

            transcript_history = self.state.transcript_candidate_history.transcript_history
            transcript_lines = [
                f"H{index}: {self._summarize_words(history)} text={self._words_text(history)!r}"
                for index, history in enumerate(transcript_history)
            ]
            transcript_lines += [
                f"CUR: {self._summarize_words(transcript_words)} text={self._words_text(transcript_words)!r}",
                f"CONF: {self._summarize_words(confirmed_words)} text={self._words_text(confirmed_words)!r}",
            ]
            LOG.debug(
                "Transcript candidates:\n%s",
                os.linesep.join(transcript_lines),
            )
            LOG.debug(
                "Transcript summary: lang=%s current=%s confirmed=%s history=%s",
                self.state.last_transcribe_lang,
                self._summarize_words(transcript_words),
                self._summarize_words(confirmed_words),
                " | ".join(self._summarize_words(history) for history in transcript_history),
            )
        confirm_ms = (perf_counter() - confirm_start) * 1000.0

        unconfirmed_words = [transcript_words[i] for i in range(len(confirmed_words), len(transcript_words))]
        self.state.last_transcribe_metrics_ms = {
            "detect_ms": detect_ms,
            "transcribe_ms": transcribe_ms,
            "confirm_ms": confirm_ms,
        }
        return confirmed_words, unconfirmed_words

    def gap_language_mismatch(self, *, start_ts: int, end_ts: int, expected_lang: str, min_gap_samples: int = 4000) -> bool:
        if not expected_lang or end_ts <= start_ts:
            return False
        if end_ts - start_ts < min_gap_samples:
            return False

        probe_ts = start_ts + (end_ts - start_ts) // 2
        detected_lang, probability, _used_samples = self.guess_language(timestamp=max(0, probe_ts))
        probe_start = max(self.state.window_ts, probe_ts - 8000)
        probe_end = min(self.state.audio_ts, probe_ts + 8000)
        nearby_unconfirmed = [word for word in self.state.unconfirmed_words if word.end_ts >= probe_start and word.start_ts <= probe_end]
        mismatch = probability > 0.5 and detected_lang != expected_lang
        LOG.debug(
            "Gap probe %d-%d expected=%s detected=%s prob=%.3f mismatch=%s probe_ts=%d nearby_unconfirmed=%s",
            start_ts,
            end_ts,
            expected_lang,
            detected_lang,
            probability,
            mismatch,
            probe_ts,
            self._summarize_words(nearby_unconfirmed),
        )
        return mismatch

    def get_next_number_of_chunks(self) -> int:
        available_chunks = self.config.window_duration - float(self.state.audio_ts - self.state.window_ts) / self.config.sampling_rate

        thresholds = self.config.chunk_table

        for limit, value in thresholds:
            limit_sample = int(limit * self.config.window_duration)
            value_sample = math.ceil(value * self.config.window_duration)
            if available_chunks >= limit_sample:
                return value_sample

        # If for some reason available_chunks is less than 0, return the smallest chunk count.
        return 1

    def pretty_print_transcript(
        self,
        acknowledged_utterances: List[Utterance],
        unacknowledged_utterances: List[Utterance],
        unconfirmed_words: List[Word],
        file: TextIO = sys.stdout,
    ):
        colour_scheme_ack = COLORSCHEME_ACKNOWLEDGED if self.config.log_colours else COLORSCHEME_NONE
        colour_scheme_unack = COLORSCHEME_UNACKNOWLEDGED if self.config.log_colours else COLORSCHEME_NONE
        colour_scheme_unconfirmed = COLORSCHEME_UNCONFIRMED if self.config.log_colours else COLORSCHEME_NONE
        formatter: TranscriptFormatter = TranscriptFormatter(
            speaker_style=SpeakerStyle.always,
            timestamp_style=TimestampStyle.range,
            probability_style=ProbabilityStyle.word,
            language_style=LanguageStyle.always,
        )
        prefix = (
            f"[{samples_to_seconds(self.state.window_ts)}/"
            f"{samples_to_seconds(self.state.audio_ts - self.state.acknowledged_ts)}/"
            f"{samples_to_seconds(self.state.audio_ts - self.state.confirmed_ts)}]"
        )
        file.write(prefix + (Fore.LIGHTGREEN_EX if self.config.log_colours else ""))
        for u in acknowledged_utterances:
            file.write(formatter.format_utterance(utterance=u, colours=colour_scheme_ack).decode("utf-8"))
        for u in unacknowledged_utterances:
            file.write(formatter.format_utterance(utterance=u, colours=colour_scheme_unack).decode("utf-8"))
        if len(unconfirmed_words) > 0:
            file.write(
                formatter.format_utterance(
                    utterance=Utterance.from_words(utterance_id=self.state.utterance_id.next(), words=unconfirmed_words),
                    colours=colour_scheme_unconfirmed,
                ).decode("utf-8")
            )
        file.write(os.linesep)
        file.flush()

    def acknowledge_utterances(
        self,
        utterances: List[Utterance],
        min_ack_duration=16000,
        min_unack_duration=16000,
    ) -> Tuple[List[Utterance], List[Utterance]]:
        if len(utterances) == 0:
            return [], utterances

        # check if the last utterance is complete
        last_valid_utterance_index = len(utterances) - 1
        valid_endings = tuple(APPEND_PUNCTUATIONS)
        if not utterances[last_valid_utterance_index].text.endswith(valid_endings):
            last_valid_utterance_index = last_valid_utterance_index - 1

        if last_valid_utterance_index < 0:
            return [], utterances

        start_ts = utterances[0].start_ts
        max_duration = self.state.audio_ts - self.state.window_ts - min_unack_duration

        duration = 0
        index = 0
        while duration < min_ack_duration and index <= last_valid_utterance_index:
            new_duration = utterances[index].end_ts - start_ts
            if new_duration > max_duration:
                break
            duration = new_duration
            index += 1

        return utterances[0:index], utterances[index:]

    def get_speaker_at(self, time: float, diarization: Optional[Annotation]) -> Optional[str]:
        if diarization is None:
            return None

        for turn, _track, speaker in diarization.itertracks(yield_label=True):  # pyright: ignore[reportAssignmentType]
            if turn.end < time:
                continue
            if turn.start > time:
                break
            return str(speaker)
        return None

    def get_speaker_before(self, time: float, diarization: Annotation) -> Optional[str]:
        last_speaker = None
        for turn, _track, speaker in diarization.itertracks(yield_label=True):  # pyright: ignore[reportAssignmentType]
            if turn.end < time:
                last_speaker = speaker
                continue
            if turn.start > time:
                break
        return str(last_speaker)

    def get_speaker_after(self, time: float, diarization: Annotation) -> Optional[str]:
        for turn, _track, speaker in diarization.itertracks(yield_label=True):  # pyright: ignore[reportAssignmentType]
            if turn.start > time:
                return str(speaker)
        return None

    def assign_speaker(self, utterance: Utterance, diarization: Optional[Annotation]) -> Optional[str]:
        if diarization is None:
            return None

        start = utterance.start_ts / 16000.0
        end = utterance.end_ts / 16000.0
        duration = end - start
        samples: List[float]
        if duration < 1:
            samples = [start + duration * 0.5]
        elif duration < 4:
            samples = [
                start + duration * 0.25,
                start + duration * 0.5,
                start + duration * 0.75,
            ]
        else:
            samples = [
                start + duration * 0.2,
                start + duration * 0.3,
                start + duration * 0.4,
                start + duration * 0.5,
                start + duration * 0.6,
                start + duration * 0.7,
                start + duration * 0.8,
            ]

        votes = {}
        speaker: Optional[str]
        for sample in samples:
            speaker: Optional[str] = self.get_speaker_at(time=sample, diarization=diarization)
            if speaker:
                if speaker in votes:
                    votes[speaker] = votes[speaker] + 1
                else:
                    votes[speaker] = 1

        if len(votes) != 0:
            best_votes = 0
            best_speaker = ""
            for speaker, vote in votes.items():
                if vote > best_votes:
                    best_speaker = speaker
                    best_votes = vote
            return best_speaker

        speaker = self.get_speaker_before(time=start + duration * 0.5, diarization=diarization)
        if speaker:
            return speaker

        speaker = self.get_speaker_after(time=start + duration * 0.5, diarization=diarization)
        if speaker:
            return speaker

        return None

    def process_audio_window(self, audio_stream: AudioStream) -> Generator[Tuple[Utterance, List[Utterance], List[Word]], None, None]:
        while True:
            iteration_start = perf_counter()
            initial_window_ts = self.state.window_ts
            initial_confirmed_ts = self.state.confirmed_ts
            initial_acknowledged_ts = self.state.acknowledged_ts
            # minimum number of samples to attempt transcription
            min_audio_duration_samples = 16000
            min_speech_duration_ms = 500
            utterances = []
            pass_metrics = {
                "detect_ms": 0.0,
                "transcribe_ms": 0.0,
                "confirm_ms": 0.0,
                "sentence_ms": 0.0,
                "acknowledge_ms": 0.0,
                "speaker_ms": 0.0,
                "output_ms": 0.0,
                "advance_ms": 0.0,
            }
            enable_vad = True
            if enable_vad and self.state.skip_silences:
                self.state.skip_silences = False
                next_ts = self.state.audio_ts
                if len(self.state.unacknowledged_utterances) > 0:
                    next_ts = min(next_ts, self.state.unacknowledged_utterances[0].start_ts)
                if len(self.state.unconfirmed_words) > 0:
                    next_ts = min(next_ts, self.state.unconfirmed_words[0].start_ts)
                self.skip_leading_silence(min_speech_duration_ms=min_speech_duration_ms, max_skip=next_ts - self.state.window_ts)
                if self.state.audio_ts - self.state.window_ts < min_audio_duration_samples:
                    # we skipped all available audio - keep skipping silences and do nothing else for now
                    self.state.skip_silences = True

            if self.state.audio_ts - self.state.window_ts < min_audio_duration_samples:
                return

            if self.config.debug and self.config.working_dir:
                self.dump_window_to_file(filename=f"{self.state.working_prefix_no_ext}-debug_window.wav")  # Dump current window for debugging

            confirmed_words, unconfirmed_words = self.transcribe_window()
            pass_metrics.update(self.state.last_transcribe_metrics_ms)
            self.state.unconfirmed_words = unconfirmed_words
            if len(confirmed_words) > 0:
                sentence_start = perf_counter()
                utterances = self.words_to_sentences(
                    word_tokenizer=BoundedSentenceTokenizer(other_tokenizer=self.models.sentence_tokenizer),
                    window_words=confirmed_words,
                    id_provider=self.state.utterance_id,
                )
                pass_metrics["sentence_ms"] += (perf_counter() - sentence_start) * 1000.0

                acknowledge_start = perf_counter()
                acknowledged_utterances, confirmed_utterances = self.acknowledge_utterances(utterances=utterances)
                pass_metrics["acknowledge_ms"] += (perf_counter() - acknowledge_start) * 1000.0

                acknowledged_to_commit = acknowledged_utterances[:1]
                deferred_acknowledged = acknowledged_utterances[1:]
                LOG.debug(
                    "Ack candidates: lang=%s anchor=%d acknowledged=%s commit=%s defer=%s confirmed=%s unconfirmed=%s",
                    self.state.last_transcribe_lang,
                    max(self.state.window_ts, self.state.acknowledged_ts),
                    self._summarize_utterances(acknowledged_utterances),
                    self._summarize_utterances(acknowledged_to_commit),
                    self._summarize_utterances(deferred_acknowledged),
                    self._summarize_utterances(confirmed_utterances),
                    self._summarize_words(unconfirmed_words),
                )

                if len(acknowledged_to_commit) > 0:
                    first_candidate = acknowledged_to_commit[0]
                    current_anchor_ts = max(self.state.window_ts, self.state.acknowledged_ts)
                    if self.gap_language_mismatch(
                        start_ts=current_anchor_ts,
                        end_ts=first_candidate.start_ts,
                        expected_lang=self.state.last_transcribe_lang,
                    ):
                        conservative_skip_samples = 100 * 16000 // 1000
                        if conservative_skip_samples > 0:
                            self.state.advance_audio_window(conservative_skip_samples)
                        self.state.skip_silences = True
                        LOG.debug(
                            "Blocked first candidate due to gap mismatch: lang=%s anchor=%d candidate=%s",
                            self.state.last_transcribe_lang,
                            current_anchor_ts,
                            self._summarize_utterance(first_candidate),
                        )
                        break

                if audio_stream.diarization:
                    speaker_start = perf_counter()
                    for acknowledged_utterance in acknowledged_to_commit:
                        acknowledged_utterance.speaker = self.assign_speaker(acknowledged_utterance, audio_stream.diarization)
                    pass_metrics["speaker_ms"] += (perf_counter() - speaker_start) * 1000.0

                previous_pending = list(self.state.unacknowledged_utterances)
                self.state.acknowledged_utterances += acknowledged_to_commit
                self.state.unacknowledged_utterances = deferred_acknowledged + confirmed_utterances
                LOG.debug(
                    "Pending replace: lang=%s prev=%s new=%s",
                    self.state.last_transcribe_lang,
                    self._summarize_utterances(previous_pending),
                    self._summarize_utterances(self.state.unacknowledged_utterances),
                )

                for i, utterance in enumerate(acknowledged_to_commit):
                    result: TranscriptionWindowResult[Utterance, Word] = TranscriptionWindowResult(
                        utterance=utterance,
                        unacknowledged=acknowledged_to_commit[i + 1 :] + deferred_acknowledged + confirmed_utterances,
                        unconfirmed_words=unconfirmed_words,
                    )
                    yield result.as_tuple()

                if len(acknowledged_to_commit) > 0:
                    for u in acknowledged_to_commit:
                        self.state.acknowledged_words += u.words

                    # utterances are split at short pauses, advance a bit to avoid repeating the last word
                    # but not too much as to skip the first word of the next utterance
                    utterance_padding_ms = 100
                    utterance_padding_samples = utterance_padding_ms * 16000 // 1000
                    accepted_end_ts = acknowledged_to_commit[-1].end_ts
                    skip_to = accepted_end_ts + utterance_padding_samples

                    next_visible_ts = None
                    if len(self.state.unacknowledged_utterances) > 0:
                        next_visible_ts = self.state.unacknowledged_utterances[0].start_ts
                    if len(self.state.unconfirmed_words) > 0:
                        next_unconfirmed_ts = self.state.unconfirmed_words[0].start_ts
                        if next_visible_ts is None or next_unconfirmed_ts < next_visible_ts:
                            next_visible_ts = next_unconfirmed_ts

                    if next_visible_ts is not None and self.gap_language_mismatch(
                        start_ts=accepted_end_ts,
                        end_ts=next_visible_ts,
                        expected_lang=self.state.last_transcribe_lang,
                    ):
                        skip_to = accepted_end_ts
                        LOG.debug(
                            "Suppressing post-ack padding due to gap mismatch: lang=%s accepted=%s next_visible=%d",
                            self.state.last_transcribe_lang,
                            self._summarize_utterance(acknowledged_to_commit[-1]),
                            next_visible_ts,
                        )

                    # try to skip ahead, but don't skip beyond the next detected word
                    if len(self.state.unacknowledged_utterances) > 0 and skip_to > self.state.unacknowledged_utterances[0].start_ts:
                        skip_to = self.state.unacknowledged_utterances[0].start_ts
                    if len(self.state.unconfirmed_words) > 0 and skip_to > self.state.unconfirmed_words[0].start_ts:
                        skip_to = self.state.unconfirmed_words[0].start_ts

                    # Never acknowledge beyond audio we've actually ingested
                    self.state.acknowledged_ts = min(skip_to, self.state.audio_ts)
                    self.state.skip_silences = True
                    LOG.debug(
                        "Committed utterance: lang=%s utterance=%s next_pending=%s next_unconfirmed=%s acknowledged_ts=%d",
                        self.state.last_transcribe_lang,
                        self._summarize_utterance(acknowledged_to_commit[-1]),
                        self._summarize_utterances(self.state.unacknowledged_utterances),
                        self._summarize_words(self.state.unconfirmed_words),
                        self.state.acknowledged_ts,
                    )
            else:
                acknowledged_utterances = []
                confirmed_utterances = []

            output_start = perf_counter()
            outstr = StringIO()
            self.pretty_print_transcript(
                acknowledged_utterances=[],
                unacknowledged_utterances=confirmed_utterances,
                unconfirmed_words=unconfirmed_words,
                file=outstr,
            )
            LOG.debug(outstr.getvalue())
            pass_metrics["output_ms"] += (perf_counter() - output_start) * 1000.0

            self.state.acknowledged_words = WhisperHistory.advance_transcript(
                timestamp=self.state.window_ts, transcript=self.state.acknowledged_words
            )

            if self.state.acknowledged_ts > self.state.window_ts:
                advance_start = perf_counter()
                shift_amount = self.state.acknowledged_ts - self.state.window_ts
                self.state.advance_audio_window(shift_amount)
                self.state.skip_silences = True
                pass_metrics["advance_ms"] += (perf_counter() - advance_start) * 1000.0

            progressed_samples = max(0, self.state.window_ts - initial_window_ts)
            confirmed_delta_s = samples_to_seconds(max(0, self.state.confirmed_ts - initial_confirmed_ts))
            acknowledged_delta_s = samples_to_seconds(max(0, self.state.acknowledged_ts - initial_acknowledged_ts))
            total_pass_ms = (perf_counter() - iteration_start) * 1000.0
            pass_metrics["total_ms"] = total_pass_ms
            self.state.timing_totals_ms["detect"] += pass_metrics["detect_ms"]
            self.state.timing_totals_ms["transcribe"] += pass_metrics["transcribe_ms"]
            self.state.timing_totals_ms["confirm"] += pass_metrics["confirm_ms"]
            self.state.timing_totals_ms["sentence"] += pass_metrics["sentence_ms"]
            self.state.timing_totals_ms["acknowledge"] += pass_metrics["acknowledge_ms"]
            self.state.timing_totals_ms["speaker"] += pass_metrics["speaker_ms"]
            self.state.timing_totals_ms["output"] += pass_metrics["output_ms"]
            self.state.timing_totals_ms["advance"] += pass_metrics["advance_ms"]
            self.state.timing_totals_ms["total"] += pass_metrics["total_ms"]
            LOG.info(
                (
                    "Pass timing: total=%.1fms detect=%.1fms transcribe=%.1fms confirm=%.1fms "
                    "sentence=%.1fms acknowledge=%.1fms speaker=%.1fms output=%.1fms advance=%.1fms "
                    "confirmed_delta=%.3fs ack_delta=%.3fs"
                ),
                pass_metrics["total_ms"],
                pass_metrics["detect_ms"],
                pass_metrics["transcribe_ms"],
                pass_metrics["confirm_ms"],
                pass_metrics["sentence_ms"],
                pass_metrics["acknowledge_ms"],
                pass_metrics["speaker_ms"],
                pass_metrics["output_ms"],
                pass_metrics["advance_ms"],
                confirmed_delta_s,
                acknowledged_delta_s,
            )
            if progressed_samples > 0:
                iteration_elapsed_seconds = max(perf_counter() - iteration_start, 1e-9)
                iteration_progress_seconds = samples_to_seconds(progressed_samples)
                iteration_speed_multiplier = iteration_progress_seconds / iteration_elapsed_seconds
                total_elapsed_seconds = max(perf_counter() - self.state.processing_started_at, 1e-9)
                total_progressed_samples = max(0, self.state.window_ts - self.state.processing_started_window_ts)
                total_progress_seconds = samples_to_seconds(total_progressed_samples)
                average_speed_multiplier = total_progress_seconds / total_elapsed_seconds
                total_samples = self._get_total_samples(audio_stream)
                current_seconds = samples_to_seconds(self.state.window_ts)
                if total_samples is not None and total_samples > 0:
                    total_seconds = samples_to_seconds(total_samples)
                    progress_percent = 100.0 * self.state.window_ts / total_samples
                    LOG.info(
                        "Clip progress %.3fs/%.3fs (%.1f%%) speed=%.2fx avg=%.2fx",
                        current_seconds,
                        total_seconds,
                        progress_percent,
                        iteration_speed_multiplier,
                        average_speed_multiplier,
                    )
                else:
                    LOG.info(
                        "Clip progress %.3fs speed=%.2fx avg=%.2fx",
                        current_seconds,
                        iteration_speed_multiplier,
                        average_speed_multiplier,
                    )

            if len(acknowledged_utterances) <= 1:
                break

    def capture_audio(self, audio_source: AudioStream):
        if not audio_source.has_more():
            return False

        next_chunk = self.get_next_number_of_chunks()
        LOG.debug(f"capturing {next_chunk} chunks of audio")
        audio_array = audio_source.next_chunk(next_chunk)
        self.state.append_audio_to_window(audio_array)
        return True

    def flush_overflowing_utterances(self, diarization: Optional[Annotation]) -> Generator[Tuple[Utterance, List[Utterance], List[Word]], None, None]:
        # As the attention window advances, we may not be able to acknowledge
        # all utterances and words; When they fall behind, the best we can do
        # is return them as acknowledge.

        # First, try to acknowledge "full" utterances that may not have been acknowledged
        while len(self.state.unacknowledged_utterances) > 0:
            # Stop when we reached the window_ts
            # Note that the == case is frequent, whisper can generate several words with timestamp 0 -> 0
            if self.state.unacknowledged_utterances[0].end_ts >= self.state.window_ts:
                break
            utterance = self.state.unacknowledged_utterances.pop(0)
            utterance.speaker = self.assign_speaker(utterance, diarization)
            self.state.acknowledged_words += utterance.words
            result: TranscriptionWindowResult[Utterance, Word] = TranscriptionWindowResult(
                utterance=utterance,
                unacknowledged=self.state.unacknowledged_utterances,
                unconfirmed_words=self.state.unconfirmed_words,
            )
            yield result.as_tuple()

        # If an utterance crosses the boundary, acknowledge the whole utterance so the
        # next pass can restart from a clean utterance boundary instead of mid-utterance.
        if len(self.state.unacknowledged_utterances) > 0:
            crossing_utterance = self.state.unacknowledged_utterances[0]
            if crossing_utterance.start_ts < self.state.window_ts:
                utterance = self.state.unacknowledged_utterances.pop(0)
                utterance.speaker = self.assign_speaker(utterance, diarization)
                self.state.acknowledged_words += utterance.words
                result: TranscriptionWindowResult[Utterance, Word] = TranscriptionWindowResult(
                    utterance=utterance,
                    unacknowledged=self.state.unacknowledged_utterances,
                    unconfirmed_words=self.state.unconfirmed_words,
                )
                yield result.as_tuple()

        # If there are unconfirmed words falling behind, acknowledge them
        flushed_utterances_words = []
        while len(self.state.unconfirmed_words) > 0:
            # Stop when we reached the window_ts
            # Note that the == case is frequent, whisper can generate several words with timestamp 0 -> 0
            if self.state.unconfirmed_words[0].end_ts >= self.state.window_ts:
                break
            flushed_word = self.state.unconfirmed_words.pop(0)
            flushed_utterances_words.append(flushed_word)

        if len(flushed_utterances_words) > 0:
            utterance = Utterance.from_words(utterance_id=self.state.utterance_id.next(), words=flushed_utterances_words)
            utterance.speaker = self.assign_speaker(utterance, diarization)
            self.state.acknowledged_words += flushed_utterances_words
            result: TranscriptionWindowResult[Utterance, Word] = TranscriptionWindowResult(
                utterance=utterance,
                unacknowledged=self.state.unacknowledged_utterances,
                unconfirmed_words=self.state.unconfirmed_words,
            )
            yield result.as_tuple()

    def transcribe_naive(
        self, audio_stream: AudioStream, working_prefix_no_ext: str = "out"
    ) -> Generator[Tuple[Utterance, List[Utterance], List[Word]], None, None]:
        self.state = State(self.config, working_prefix_no_ext=working_prefix_no_ext)
        if audio_stream.start_offset != 0:
            self.state.window_ts = audio_stream.start_offset
            self.state.audio_ts = audio_stream.start_offset

        self._emit_status(StatusUpdate(state="transcribing"))
        full_audio = self._read_full_audio(audio_stream)
        if full_audio.size == 0:
            return

        self.state.audio_ts = self.state.window_ts + len(full_audio)

        detect_start = perf_counter()
        lang_result = self._detect_language_for_audio(
            audio=full_audio,
            window_ts=self.state.window_ts,
            audio_ts=self.state.audio_ts,
            timestamp=max(0, self.state.window_ts),
        )
        lang = lang_result.language
        self.state.last_transcribe_lang = lang
        detect_ms = (perf_counter() - detect_start) * 1000.0

        whisper_prompt = self.config.whisper_prompts[lang] if lang in self.config.whisper_prompts else self.config.whisper_prompts["en"]
        transcribe_start = perf_counter()
        transcript_words = self.models.transcriber.transcribe(
            audio=full_audio,
            lang=lang,
            prompt=whisper_prompt,
            prefix="",
            window_ts=self.state.window_ts,
            audio_ts=self.state.audio_ts,
            whisper_beam_size=self.config.whisper_beam_size,
            whisper_best_of=self.config.whisper_best_of,
            whisper_patience=self.config.whisper_patience,
            whisper_temperatures=self.config.whisper_temperatures,
        )
        transcribe_ms = (perf_counter() - transcribe_start) * 1000.0
        self.state.timing_totals_ms["detect"] += detect_ms
        self.state.timing_totals_ms["transcribe"] += transcribe_ms

        if len(transcript_words) == 0:
            self.state.timing_totals_ms["total"] += detect_ms + transcribe_ms
            return

        sentence_start = perf_counter()
        utterances = self.words_to_sentences(
            word_tokenizer=self.models.sentence_tokenizer,
            window_words=transcript_words,
            id_provider=self.state.utterance_id,
        )
        sentence_ms = (perf_counter() - sentence_start) * 1000.0
        self.state.timing_totals_ms["sentence"] += sentence_ms

        speaker_ms = 0.0
        if audio_stream.diarization:
            speaker_start = perf_counter()
            for utterance in utterances:
                utterance.speaker = self.assign_speaker(utterance, audio_stream.diarization)
            speaker_ms = (perf_counter() - speaker_start) * 1000.0
            self.state.timing_totals_ms["speaker"] += speaker_ms

        self.state.timing_totals_ms["total"] += detect_ms + transcribe_ms + sentence_ms + speaker_ms
        self._emit_transcribing_progress(audio_stream)

        for i, utterance in enumerate(utterances):
            remaining = utterances[i + 1 :]
            self._emit_utterance_status(utterance=utterance, unacknowledged=remaining, unconfirmed=[])
            yield utterance, remaining, []

    def transcribe(
        self, audio_stream: AudioStream, working_prefix_no_ext: str = "out"
    ) -> Generator[Tuple[Utterance, List[Utterance], List[Word]], None, None]:
        if not self.config.code_switching:
            yield from self.transcribe_naive(audio_stream=audio_stream, working_prefix_no_ext=working_prefix_no_ext)
            return

        self.state = State(self.config, working_prefix_no_ext=working_prefix_no_ext)
        try:
            if audio_stream.start_offset != 0:
                self.state.window_ts = audio_stream.start_offset
                self.state.audio_ts = audio_stream.start_offset

            STATUS_LOG.info("Starting main loop for audio transcription.")
            self._emit_status(StatusUpdate(state="transcribing"))
            while True:
                has_more_audio = self.capture_audio(audio_source=audio_stream)
                had_utterances = False

                # capture any utterance that slipped out of the current window
                for utterance, unacknowledged, unconfirmed in self.flush_overflowing_utterances(diarization=audio_stream.diarization):
                    self._emit_utterance_status(
                        utterance=utterance,
                        unacknowledged=unacknowledged,
                        unconfirmed=unconfirmed,
                    )
                    yield utterance, unacknowledged, unconfirmed

                # attempt to acknowledge new utterances from current window
                for utterance, unacknowmedged, unconfirmed in self.process_audio_window(audio_stream=audio_stream):
                    had_utterances = True
                    self._emit_utterance_status(
                        utterance=utterance,
                        unacknowledged=unacknowmedged,
                        unconfirmed=unconfirmed,
                    )
                    yield utterance, unacknowmedged, unconfirmed

                self._emit_transcribing_progress(audio_stream)

                if not had_utterances and not has_more_audio:
                    break

        except KeyboardInterrupt:
            LOG.warning("KeyboardInterrupt detected, stopping transcription.")
            LOG.debug("Stopping...")
        # pylint: disable=broad-exception-caught
        except Exception as e:
            LOG.error(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
            LOG.debug("Stopping...")
        finally:
            LOG.info(
                (
                    "Pipeline totals: detect=%.1fms transcribe=%.1fms confirm=%.1fms sentence=%.1fms "
                    "acknowledge=%.1fms speaker=%.1fms output=%.1fms advance=%.1fms total=%.1fms"
                ),
                self.state.timing_totals_ms["detect"],
                self.state.timing_totals_ms["transcribe"],
                self.state.timing_totals_ms["confirm"],
                self.state.timing_totals_ms["sentence"],
                self.state.timing_totals_ms["acknowledge"],
                self.state.timing_totals_ms["speaker"],
                self.state.timing_totals_ms["output"],
                self.state.timing_totals_ms["advance"],
                self.state.timing_totals_ms["total"],
            )
            for i, utterance in enumerate(self.state.unacknowledged_utterances):
                utterance.speaker = self.assign_speaker(utterance, audio_stream.diarization)
                LOG.debug("Finalizing leftover pending utterance: %s", self._summarize_utterance(utterance))
                self._emit_utterance_status(
                    utterance=utterance,
                    unacknowledged=self.state.unacknowledged_utterances[i + 1 :],
                    unconfirmed=self.state.unconfirmed_words,
                )
                yield utterance, self.state.unacknowledged_utterances[i + 1 :], self.state.unconfirmed_words

            if len(self.state.unconfirmed_words) > 0:
                unconfirmed_utterance: Utterance = Utterance.from_words(
                    utterance_id=self.state.utterance_id.next(), words=self.state.unconfirmed_words
                )
                unconfirmed_utterance.speaker = self.assign_speaker(unconfirmed_utterance, audio_stream.diarization)
                LOG.debug("Finalizing leftover unconfirmed words: %s", self._summarize_utterance(unconfirmed_utterance))
                self._emit_utterance_status(
                    utterance=unconfirmed_utterance,
                    unacknowledged=[],
                    unconfirmed=[],
                )
                yield unconfirmed_utterance, [], []


def execute(
    *,
    config: Config,
    source_path: str,
    audio_sources: List[AudioSource],
    write_config: TranscriptWriterConfig,
    output_formats: List[str],
    output_prefix_no_ext: str,
    working_prefix_no_ext: str,
    eval_file: Optional[str],
    status_hook: Optional[StatusHook] = None,
):
    all_utterances: List[Utterance] = []
    transcriber = Verbatim(config, status_hook=status_hook)
    stdout_enabled = True
    for idx, audio_source in enumerate(audio_sources):
        STATUS_LOG.info("Transcribing from audio source: %s", audio_source.source_name)
        # When multiple sources are present (e.g., per-channel VTTM), isolate per-source outputs to avoid clobbering.
        per_source_prefix = output_prefix_no_ext
        if len(audio_sources) > 1:
            suffix = getattr(audio_source, "file_id", None) or f"src{idx}"
            per_source_prefix = f"{output_prefix_no_ext}-{suffix}"
        writer = configure_writers(
            write_config,
            output_formats=output_formats,
            original_audio_file=audio_source.source_name,
            output_prefix_no_ext=per_source_prefix,
            stdout_enabled=stdout_enabled,
        )
        writer.open()
        with audio_source.open() as audio_stream:
            for utterance, unacknowledged, unconfirmed in transcriber.transcribe(
                audio_stream=audio_stream, working_prefix_no_ext=working_prefix_no_ext
            ):
                writer.write(
                    utterance=utterance,
                    unacknowledged_utterance=unacknowledged,
                    unconfirmed_words=unconfirmed,
                )
                all_utterances.append(utterance)
        writer.close()
        STATUS_LOG.info("Done transcribing from audio source: %s", audio_source.source_name)

    if len(audio_sources) > 1:
        sorted_utterances: List[Utterance] = sorted(all_utterances, key=lambda x: x.start_ts)
        writer = configure_writers(
            write_config,
            output_formats=output_formats,
            original_audio_file=source_path,
            output_prefix_no_ext=output_prefix_no_ext,
            stdout_enabled=stdout_enabled,
        )
        writer.open()
        for sorted_utterance in sorted_utterances:
            writer.write(utterance=sorted_utterance)
        writer.close()

    if eval_file:
        LOG.info("Lazy-loading evaluation metrics.")
        from .eval.compare import compute_metrics  # pylint: disable=import-outside-toplevel

        sorted_utterances: List[Utterance] = sorted(all_utterances, key=lambda x: x.start_ts)
        ref_utterances: List[Utterance] = read_utterances(eval_file)
        metrics = compute_metrics(sorted_utterances, ref_utterances)
        print(metrics)

import logging
import math
import os
import sys
import traceback
import wave
from dataclasses import dataclass, field
from io import StringIO
from typing import Generator, List, Optional, TextIO, Tuple

import numpy as np
from colorama import Fore
from numpy.typing import NDArray
from pyannote.core.annotation import Annotation

from .audio.audio import samples_to_seconds
from .audio.sources.audiosource import AudioSource, AudioStream
from .config import Config
from .eval.compare import compute_metrics
from .models import Models
from .transcript.format.factory import configure_writers
from .transcript.format.json import read_utterances
from .transcript.format.txt import (
    COLORSCHEME_ACKNOWLEDGED,
    COLORSCHEME_UNACKNOWLEDGED,
    COLORSCHEME_UNCONFIRMED,
    TranscriptFormatter,
)
from .transcript.format.writer import (
    LanguageStyle,
    ProbabilityStyle,
    SpeakerStyle,
    TimestampStyle,
    TranscriptWriter,
    TranscriptWriterConfig,
)
from .transcript.idprovider import CounterIdProvider, IdProvider
from .transcript.sentences import SentenceTokenizer, SilenceSentenceTokenizer
from .transcript.words import Utterance, Word

# pylint: disable=unused-import
from .voices.transcribe.transcribe import APPEND_PUNCTUATIONS, PREPEND_PUNCTUATIONS

# Configure logger
LOG = logging.getLogger(__name__)


@dataclass
class WhisperHistory:
    size: int = 12
    transcript_history: List[List[Word]] = field(default_factory=list)

    def add(self, transcription: List[Word]):
        self.transcript_history.append(transcription)
        if len(self.transcript_history) > self.size:
            self.transcript_history.pop(0)

    @staticmethod
    def advance_transcript(timestamp: int, transcript: List[Word]) -> List[Word]:
        return [w for w in transcript if w.end_ts >= timestamp]

    def advance(self, timestamp: int):
        self.transcript_history = [self.advance_transcript(timestamp, transcript) for transcript in self.transcript_history]
        self.transcript_history = [h for h in self.transcript_history if len(h) > 0]

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
                LOG.debug(f"Confirming word '{transcript[p_index].word}'")
                confirmed_words += [current_words[c_index]]
                p_index += 1
                c_index += 1
            else:
                break
        if len(confirmed_words) > 0:
            LOG.debug(f"CONFIRMED CANDIDATE: {''.join([w.word for w in confirmed_words])}")
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
    transcript_candidate_history: WhisperHistory
    rolling_window: RollingWindow
    acknowledged_words: List[Word]
    unconfirmed_words: List[Word]
    acknowledged_utterances: List[Utterance]
    unacknowledged_utterances: List[Utterance]
    skip_silences: bool = True
    speaker_embeddings: Optional[List] = None
    working_prefix_no_ext: str = "out"

    def __init__(self, config: Config, working_prefix_no_ext: str = "out"):
        self.config = config
        self.confirmed_ts = -1
        self.acknowledged_ts = -1
        self.window_ts = 0  # Window timestamp in samples
        self.audio_ts = 0  # Global timestamp in samples
        self.transcript_candidate_history = WhisperHistory()
        self.acknowledged_words = []
        self.unconfirmed_words = []
        self.acknowledged_utterances = []
        self.unacknowledged_utterances = []
        self.speaker_embeddings = []
        self.working_prefix_no_ext = working_prefix_no_ext
        self.utterance_id: IdProvider = CounterIdProvider(prefix="utt")

        window_size = self.config.sampling_rate * self.config.window_duration  # Total samples in 30 seconds
        self.rolling_window = RollingWindow(window_size=window_size, dtype=np.float32)  # Initialize empty rolling window

    def advance_audio_window(self, offset: int):
        if offset <= 0:
            return
        # Do not advance beyond available audio; keep window_ts <= audio_ts
        max_advance = max(0, self.audio_ts - self.window_ts)
        if offset > max_advance:
            LOG.warning(
                f"advance_audio_window requested {samples_to_seconds(offset)}s but only "
                f"{samples_to_seconds(max_advance)}s available; clamping."
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

        LOG.info(f"Window now has {samples_to_seconds(self.audio_ts - self.window_ts)} seconds of audio.")

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

    def __init__(self, config: Config, models=None):
        self.config = config
        self.state = State(config)
        if models is None:
            models = Models(device=config.device, whisper_model_size=config.whisper_model_size, stream=config.stream)
        self.models = models

    def skip_leading_silence(self, max_skip: int, min_speech_duration_ms: int = 500) -> int:
        min_speech_duration_ms = 750
        min_speech_duration_samples = 16000 * min_speech_duration_ms // 1000
        audio_samples = self.state.audio_ts - self.state.window_ts
        voice_segments = self.models.vad.find_activity(
            audio=self.state.rolling_window.array[0:audio_samples],
            min_speech_duration_ms=min_speech_duration_ms,
        )
        LOG.debug(f"Voice segments: {voice_segments}")
        if len(voice_segments) == 0:
            # preserve a bit of audio at the end that may have been too short just because it is truncated
            padding = min_speech_duration_samples
            remaining_audio = self.state.audio_ts - self.state.window_ts - padding
            advance_to = remaining_audio
            # pylint: disable=consider-using-min-builtin
            if max_skip < advance_to:
                advance_to = max_skip
            LOG.info(
                f"Skipping silences between {samples_to_seconds(self.state.window_ts):.2f}"
                f" and {samples_to_seconds(self.state.window_ts + advance_to):.2f}"
            )

            self.state.advance_audio_window(remaining_audio)
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
            LOG.info(
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

        def remove_spaces_and_punctuation(string: str) -> str:
            return string.translate(str.maketrans("", "", " " + PREPEND_PUNCTUATIONS + APPEND_PUNCTUATIONS))

        # For each sentence, we want to collect a sublist of VerbatimWord
        for sentence in sentences:
            # igonre punctuation and spaces
            sentence = remove_spaces_and_punctuation(sentence)
            sentence_length = len(sentence.strip())
            target_end = current_char_index + sentence_length

            sentence_words = []
            # Accumulate words until we have accounted for all chars of this sentence
            while current_char_index < target_end:
                # Pick the next word
                w = window_words[current_word_index]
                sentence_words.append(w)
                word_text = remove_spaces_and_punctuation(w.word)
                current_char_index += len(word_text)
                current_word_index += 1

                # If we overshoot, something is wrong,
                # but given the perfect alignment we expect to land exactly on target_end
                if current_char_index > target_end:
                    raise ValueError("Mismatch in alignment between sentences and words.")

            # Now we have all the words for this sentence
            if len(sentence_words) > 0:
                result.append(Utterance.from_words(utterance_id=id_provider.next(), words=sentence_words))

        # At the end, `result` should be a List[List[VerbatimWord]]
        return result

    @staticmethod
    def words_to_sentences(word_tokenizer: SentenceTokenizer, window_words: List[Word], id_provider: IdProvider) -> list[Utterance]:
        sentences = []
        if len(window_words) == 0:
            return []

        for tok in word_tokenizer.split(words=window_words):
            sentences += [tok]

        utterances = Verbatim.align_words_to_sentences(id_provider=id_provider, sentences=sentences, window_words=window_words)
        return utterances

    def _guess_language(self, audio: NDArray, sample_offset: int, sample_duration: int, lang: List[str]) -> Tuple[str, float]:
        lang_samples = audio[sample_offset : sample_offset + sample_duration]
        LOG.info(
            f"Detecting language using samples {sample_offset}({samples_to_seconds(sample_offset)}) "
            f"to {sample_offset + sample_duration}({samples_to_seconds(sample_offset + sample_duration)})"
        )
        return self.models.transcriber.guess_language(audio=lang_samples, lang=lang)

    # returns the language guessed, the proability and the number of samples used to guess
    def guess_language(self, timestamp: int) -> Tuple[str, float, int]:
        if len(self.config.lang) == 0:
            LOG.warning("Language is not set - defaulting to english")
            return ("en", 1.0, 0)

        if len(self.config.lang) == 1:
            return (self.config.lang[0], 1.0, 0)

        lang_sample_start = max(0, timestamp - self.state.window_ts)
        available_samples = self.state.audio_ts - self.state.window_ts - lang_sample_start
        lang_samples_size = min(2 * 16000, available_samples)

        while True:
            lang, prob = self._guess_language(
                audio=self.state.rolling_window.array,
                sample_offset=lang_sample_start,
                sample_duration=lang_samples_size,
                lang=self.config.lang,
            )
            if prob > 0.5 or lang_samples_size == available_samples:
                break
            # retry with larger sample
            lang_samples_size = min(2 * lang_samples_size, available_samples)

        return (lang, prob, lang_samples_size)

    def transcribe_window(self) -> Tuple[List[Word], List[Word]]:
        LOG.info("Starting transcription of audio chunk.")
        LOG.info(f"Window Start Time: {samples_to_seconds(self.state.window_ts)}")
        LOG.info(f"Confirmed Time: {samples_to_seconds(self.state.confirmed_ts)}")
        LOG.info(f"Acknowledged Time: {samples_to_seconds(self.state.acknowledged_ts)}")
        LOG.info(f"Valid audio range: 0.0 - {samples_to_seconds(self.state.audio_ts - self.state.window_ts)}")

        acknowledged_words_in_window = WhisperHistory.advance_transcript(timestamp=self.state.window_ts, transcript=self.state.acknowledged_words)
        lang, _prob, used_samples_for_language = self.guess_language(timestamp=max(0, self.state.acknowledged_ts))
        prefix_text = ""
        for word in [w for u in self.state.unacknowledged_utterances for w in u.words]:
            if word.lang != lang:
                break
            prefix_text += word.word
        whisper_prompt = self.config.whisper_prompts[lang] if lang in self.config.whisper_prompts else self.config.whisper_prompts["en"]

        try:
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
        except RuntimeError as e:
            LOG.warning(f"Transcription failed with RuntimeError: {str(e)}. Skipping this chunk.")
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

            newline = os.linesep
            transcript_history = self.state.transcript_candidate_history.transcript_history
            LOG.debug(
                f"Transcript:\n{newline.join([''.join([w.word for w in history]) for history in transcript_history])}\n"
                f"{''.join(w.word for w in transcript_words)}\n{''.join(w.word for w in confirmed_words)}"
            )

        unconfirmed_words = [transcript_words[i] for i in range(len(confirmed_words), len(transcript_words))]
        return confirmed_words, unconfirmed_words

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
        formatter: TranscriptFormatter = TranscriptFormatter(
            speaker_style=SpeakerStyle.always,
            timestamp_style=TimestampStyle.range,
            probability_style=ProbabilityStyle.word,
            language_style=LanguageStyle.always,
        )
        file.write(
            f"[{samples_to_seconds(self.state.window_ts)}/"
            f"{samples_to_seconds(self.state.audio_ts - self.state.acknowledged_ts)}/"
            f"{samples_to_seconds(self.state.audio_ts - self.state.confirmed_ts)}]" + Fore.LIGHTGREEN_EX
        )
        for u in acknowledged_utterances:
            formatter.format_utterance(utterance=u, out=file, colours=COLORSCHEME_ACKNOWLEDGED)
        for u in unacknowledged_utterances:
            formatter.format_utterance(utterance=u, out=file, colours=COLORSCHEME_UNACKNOWLEDGED)
        if len(unconfirmed_words) > 0:
            formatter.format_utterance(
                utterance=Utterance.from_words(utterance_id=self.state.utterance_id.next(), words=unconfirmed_words),
                out=file,
                colours=COLORSCHEME_UNCONFIRMED,
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
            # minimum number of samples to attempt transcription
            min_audio_duration_samples = 16000
            min_speech_duration_ms = 500
            utterances = []
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

            if self.config.debug:
                self.dump_window_to_file(filename=f"{self.state.working_prefix_no_ext}-debug_window.wav")  # Dump current window for debugging

            confirmed_words, unconfirmed_words = self.transcribe_window()
            self.state.unconfirmed_words = unconfirmed_words
            if len(confirmed_words) > 0:
                utterances = self.words_to_sentences(
                    word_tokenizer=self.models.sentence_tokenizer,
                    window_words=confirmed_words,
                    id_provider=self.state.utterance_id,
                )

                window_duration = samples_to_seconds(self.state.audio_ts - self.state.window_ts)
                if window_duration > 25 and len(utterances) == 1:
                    utterances = self.words_to_sentences(
                        word_tokenizer=SilenceSentenceTokenizer(), window_words=confirmed_words, id_provider=self.state.utterance_id
                    )

                acknowledged_utterances, confirmed_utterances = self.acknowledge_utterances(utterances=utterances)

                if audio_stream.diarization:
                    for acknowledged_utterance in acknowledged_utterances:
                        acknowledged_utterance.speaker = self.assign_speaker(acknowledged_utterance, audio_stream.diarization)

                self.state.acknowledged_utterances += acknowledged_utterances
                self.state.unacknowledged_utterances = confirmed_utterances

                for i, utterance in enumerate(acknowledged_utterances):
                    yield utterance, acknowledged_utterances[i + 1 :] + confirmed_utterances, unconfirmed_words

                if len(acknowledged_utterances) > 0:
                    for u in acknowledged_utterances:
                        self.state.acknowledged_words += u.words

                    # utterances are split at short pauses, advance a bit to avoid repeating the last word
                    # but not too much as to skip the first word of the next utterance
                    utterance_padding_ms = 100
                    utterance_padding_samples = utterance_padding_ms * 16000 // 1000
                    skip_to = acknowledged_utterances[-1].end_ts + utterance_padding_samples

                    # try to skip ahead, but don't skip beyond the next detected word
                    if len(self.state.unacknowledged_utterances) > 0 and skip_to > self.state.unacknowledged_utterances[0].start_ts:
                        skip_to = self.state.unacknowledged_utterances[0].start_ts
                    if len(self.state.unconfirmed_words) > 0 and skip_to > self.state.unconfirmed_words[0].start_ts:
                        skip_to = self.state.unconfirmed_words[0].start_ts

                    # Never acknowledge beyond audio we've actually ingested
                    self.state.acknowledged_ts = min(skip_to, self.state.audio_ts)
                    self.state.skip_silences = True
            else:
                acknowledged_utterances = []
                confirmed_utterances = []

            outstr = StringIO()
            self.pretty_print_transcript(
                acknowledged_utterances=[],
                unacknowledged_utterances=confirmed_utterances,
                unconfirmed_words=unconfirmed_words,
                file=outstr,
            )
            LOG.info(outstr.getvalue())

            self.state.acknowledged_words = WhisperHistory.advance_transcript(
                timestamp=self.state.window_ts, transcript=self.state.acknowledged_words
            )

            if self.state.acknowledged_ts > self.state.window_ts:
                shift_amount = self.state.acknowledged_ts - self.state.window_ts
                self.state.advance_audio_window(shift_amount)
                self.state.skip_silences = True

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

        flushed_utterances = []

        # First, try to acknowledge "full" utterances that may not have been acknowledged
        while len(self.state.unacknowledged_utterances) > 0:
            # Stop when we reached the window_ts
            # Note that the == case is frequent, whisper can generate several words with timestamp 0 -> 0
            if self.state.unacknowledged_utterances[0].end_ts >= self.state.window_ts:
                break
            utterance = self.state.unacknowledged_utterances.pop(0)
            utterance.speaker = self.assign_speaker(utterance, diarization)
            yield utterance, self.state.unacknowledged_utterances, self.state.unconfirmed_words

        # If there is a partial utterance overflowing, acknowledge words from it
        if len(self.state.unacknowledged_utterances) > 0:
            flushed_utterances_words = []
            partial_utterance = self.state.unacknowledged_utterances[0]
            while len(partial_utterance.words) > 0:
                # Stop when we reached the window_ts
                # Note that the == case is frequent, whisper can generate several words with timestamp 0 -> 0
                if partial_utterance.words[0].end_ts >= self.state.window_ts:
                    break
                flushed_word = partial_utterance.words.pop(0)
                flushed_utterances_words.append(flushed_word)
                partial_utterance.start_ts = partial_utterance.words[-1].start_ts
                partial_utterance.text = "".join([w.word for w in partial_utterance.words])

            if len(flushed_utterances_words) > 0:
                utterance = Utterance.from_words(utterance_id=self.state.utterance_id.next(), words=flushed_utterances_words)
                utterance.speaker = self.assign_speaker(utterance, diarization)
                yield utterance, self.state.unacknowledged_utterances, self.state.unconfirmed_words

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
            yield utterance, self.state.unacknowledged_utterances, self.state.unconfirmed_words

        if len(flushed_utterances_words) > 0:
            flushed_utterances.append(Utterance.from_words(utterance_id=self.state.utterance_id.next(), words=flushed_utterances_words))

    def transcribe(
        self, audio_stream: AudioStream, working_prefix_no_ext: str = "out"
    ) -> Generator[Tuple[Utterance, List[Utterance], List[Word]], None, None]:
        self.state = State(self.config, working_prefix_no_ext=working_prefix_no_ext)
        try:
            if audio_stream.start_offset != 0:
                self.state.window_ts = audio_stream.start_offset
                self.state.audio_ts = audio_stream.start_offset

            LOG.info("Starting main loop for audio transcription.")
            while True:
                has_more_audio = self.capture_audio(audio_source=audio_stream)
                had_utterances = False

                # capture any utterance that slipped out of the current window
                yield from self.flush_overflowing_utterances(diarization=audio_stream.diarization)

                # attempt to acknowledge new utterances from current window
                for utterance, unacknowmedged, unconfirmed in self.process_audio_window(audio_stream=audio_stream):
                    had_utterances = True
                    yield utterance, unacknowmedged, unconfirmed

                if not had_utterances and not has_more_audio:
                    break

        except KeyboardInterrupt:
            LOG.info("KeyboardInterrupt detected, stopping transcription.")
            LOG.debug("Stopping...")
        # pylint: disable=broad-exception-caught
        except Exception as e:
            LOG.error(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
            LOG.debug("Stopping...")
        finally:
            for i, utterance in enumerate(self.state.unacknowledged_utterances):
                utterance.speaker = self.assign_speaker(utterance, audio_stream.diarization)
                yield utterance, self.state.unacknowledged_utterances[i + 1 :], self.state.unconfirmed_words

            if len(self.state.unconfirmed_words) > 0:
                unconfirmed_utterance: Utterance = Utterance.from_words(
                    utterance_id=self.state.utterance_id.next(), words=self.state.unconfirmed_words
                )
                unconfirmed_utterance.speaker = self.assign_speaker(unconfirmed_utterance, audio_stream.diarization)
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
):
    all_utterances: List[Utterance] = []
    transcriber = Verbatim(config)
    for audio_source in audio_sources:
        LOG.info(f"Transcribing from audio source: {audio_source.source_name}")
        writer: TranscriptWriter = configure_writers(
            write_config,
            output_formats=output_formats,
            original_audio_file=audio_source.source_name,
        )
        writer.open(path_no_ext=output_prefix_no_ext)
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
        LOG.info(f"Done transcribing from audio source: {audio_source.source_name}")

    if len(audio_sources) > 1:
        sorted_utterances: List[Utterance] = sorted(all_utterances, key=lambda x: x.start_ts)
        writer: TranscriptWriter = configure_writers(write_config, output_formats=output_formats, original_audio_file=source_path)
        writer.open(path_no_ext=output_prefix_no_ext)
        for sorted_utterance in sorted_utterances:
            writer.write(utterance=sorted_utterance)
        writer.close()

    if eval_file:
        sorted_utterances: List[Utterance] = sorted(all_utterances, key=lambda x: x.start_ts)
        ref_utterances: List[Utterance] = read_utterances(eval_file)
        metrics = compute_metrics(sorted_utterances, ref_utterances)
        print(metrics)

import logging
import os
import sys
import wave
import traceback
from dataclasses import dataclass, field
from io import StringIO
from typing import List, Tuple, TextIO, Generator

import numpy as np
from colorama import Fore
from pyannote.core.annotation import Annotation

from .audio.sources.audiosource import AudioSource
from .transcript.words import VerbatimWord, VerbatimUtterance
from .audio.audio import samples_to_seconds
from .config import Config
from .models import Models
from .transcript.format.txt import (
    TranscriptFormatter,
    COLORSCHEME_ACKNOWLEDGED,
    COLORSCHEME_UNACKNOWLEDGED,
    COLORSCHEME_UNCONFIRMED
)
#pylint: disable=unused-import
from .voices.transcribe.transcribe import APPEND_PUNCTUATIONS, PREPEND_PUNCTUATIONS

# Configure logger
LOG = logging.getLogger(__name__)

@dataclass
class WhisperHistory:
    size: int = 12
    transcript_history: List[List[VerbatimWord]] = field(default_factory=list)

    def add(self, transcription: List[VerbatimWord]):
        self.transcript_history.append(transcription)
        if len(self.transcript_history) > self.size:
            self.transcript_history.pop(0)

    @staticmethod
    def advance_transcript(timestamp: int, transcript: List[VerbatimWord]) -> List[VerbatimWord]:
        return [w for w in transcript if w.end_ts > timestamp]

    def advance(self, timestamp: int):
        self.transcript_history = [self.advance_transcript(timestamp, transcript) for transcript in self.transcript_history]
        self.transcript_history = [h for h in self.transcript_history if len(h) > 0]

    @staticmethod
    def confirm_transcript(
        current_words:List[VerbatimWord],
        transcript:List[VerbatimWord],
        prefix:List[VerbatimWord],
        after_ts:int
    ) -> List[VerbatimWord]:
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

    def confirm(self, current_words: List[VerbatimWord], prefix: List[VerbatimWord], after_ts: int) -> List[VerbatimWord]:
        return max((self.confirm_transcript(current_words=current_words, transcript=transcript, prefix=prefix, after_ts=after_ts)
                        for transcript in self.transcript_history),
                        key=len, default=[])

class RollingWindow:
    array:np.array
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
    rolling_window:RollingWindow
    acknowledged_words:List[VerbatimWord]
    unconfirmed_words:List[VerbatimWord]
    acknowledged_utterances:List[VerbatimUtterance]
    unacknowledged_utterances:List[VerbatimUtterance]
    skip_silences: bool = True
    speaker_embeddings:List = None


    def __init__(self, config:Config):
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

        window_size = self.config.sampling_rate * self.config.window_duration  # Total samples in 30 seconds
        self.rolling_window = RollingWindow(window_size=window_size, dtype=np.float32)  # Initialize empty rolling window

    def advance_audio_window(self, offset:int):
        if offset <= 0:
            return
        LOG.debug(f"Shifting rolling window by {offset} samples ({samples_to_seconds(offset)}s) "
                    "to offset {self.window_ts + offset} ({samples_to_seconds(self.window_ts + offset)}s).")

        self.rolling_window.array = np.roll(self.rolling_window.array, -offset)
        self.rolling_window.array[-offset:] = 0
        self.window_ts += offset

    def append_audio_to_window(self, audio_chunk:np.array):
        chunk_size = len(audio_chunk)
        window_size = len(self.rolling_window.array)
        if self.audio_ts + chunk_size <= self.window_ts + window_size:
            # Append to the current rolling window
            self.rolling_window.array[self.audio_ts - self.window_ts:self.audio_ts - self.window_ts + chunk_size] = audio_chunk
        else:
            # Shift the window and append new audio
            shift_amount = self.audio_ts + chunk_size - (self.window_ts + window_size)
            LOG.warning(f"failed to acknowledge transcription within audio buffer of {samples_to_seconds(window_size)} seconds")
            self.advance_audio_window(shift_amount)
            self.skip_silences = True
            self.rolling_window.array[-chunk_size:] = audio_chunk

        self.audio_ts += chunk_size


class Verbatim:
    state:State = None
    config:Config = None

    def __init__(self, config:Config, models=None):
        self.config = config
        self.state = State(config)
        if models is None:
            models = Models(device=config.device, stream=config.stream)
        self.models = models

        if config.start_time != 0:
            self.state.window_ts = config.start_time
            self.state.audio_ts = config.start_time

    def skip_leading_silence(self, min_speech_duration_ms:int = 500) -> int:
        min_speech_duration_ms = 750
        min_speech_duration_samples = (16000 * min_speech_duration_ms // 1000)
        audio_samples = self.state.audio_ts - self.state.window_ts
        voice_segments = self.models.vad.find_activity(audio=self.state.rolling_window.array[0:audio_samples], min_speech_duration_ms=min_speech_duration_ms)
        LOG.debug(f"Voice segments: {voice_segments}")
        if len(voice_segments) == 0:
            LOG.info(f"Skipping silences between {samples_to_seconds(self.state.window_ts):.2f} and {samples_to_seconds(self.state.audio_ts):.2f}")

            # preserve a bit of audio at the end that may have been too short just because it is truncated
            padding = min_speech_duration_samples
            remaining_audio = self.state.audio_ts - self.state.window_ts - padding
            self.state.advance_audio_window(remaining_audio)

            return self.state.audio_ts

        voice_start = voice_segments[0]['start']
        voice_end = voice_segments[0]['end']
        voice_length = voice_end - voice_start

        # skip leading silences
        if voice_start > 0:
            LOG.info(f"Skipping silences between {samples_to_seconds(self.state.window_ts):.2f} and {samples_to_seconds(self.state.window_ts + voice_start):.2f}")
            self.state.advance_audio_window(voice_start)
            return voice_length + self.state.window_ts

        return voice_length + self.state.window_ts

    def dump_window_to_file(self, filename: str = "debug_window.wav"):
        window = self.state.rolling_window.array
        LOG.debug("Dumping current window to file for debugging.")
        # pylint: disable=no-member
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.config.sampling_rate)
            wf.writeframes((window * 32768.0).astype(np.int16).tobytes())
        LOG.debug("Finished writing window to file.")


    @staticmethod
    def align_words_to_sentences(sentences: list[str], window_words: list[VerbatimWord]) -> list[VerbatimUtterance]:
        # Concatenate all sentences into a single string
        full_text_sentences = ''.join(sentences)
        # Concatenate all words from window_words
        full_text_words = ''.join(w.word for w in window_words)

        # Basic validation step (optional but recommended)
        if full_text_sentences != full_text_words:
            raise ValueError("The joined text from sentences and window_words do not match.")

        # We'll iterate through sentences and pick words until we match them character-for-character.
        result = []
        current_char_index = 0
        current_word_index = 0

        def remove_spaces_and_punctuation(string: str) -> str:
            return string.translate(str.maketrans('', '', " " + PREPEND_PUNCTUATIONS + APPEND_PUNCTUATIONS))


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
                result.append(VerbatimUtterance.from_words(sentence_words))

        # At the end, `result` should be a List[List[VerbatimWord]]
        return result


    @staticmethod
    def words_to_sentences(word_tokenizer, window_words:List[VerbatimWord]) -> list[VerbatimUtterance]:
        sentences = []
        if len(window_words) == 0:
            return []

        window_text = ''.join([w.word for w in window_words])
        for tok in word_tokenizer.split(window_text):
            sentences += [tok]

        utterances = Verbatim.align_words_to_sentences(sentences=sentences, window_words=window_words)
        return utterances

    def _guess_language(self, audio:np.array, sample_offset:int, sample_duration:int, lang:List[str]) -> Tuple[str,float]:
        lang_samples = audio[sample_offset:sample_offset + sample_duration]
        LOG.info(f"Detecting language using samples {sample_offset}({samples_to_seconds(sample_offset)}) "
                 f"to {sample_offset + sample_duration}({samples_to_seconds(sample_offset + sample_duration)})")
        return self.models.transcriber.guess_language(audio=lang_samples, lang=lang)


    def guess_language(self, timestamp:int):

        if len(self.config.lang) == 0:
            LOG.warning("Language is not set - defaulting to english")
            return "en"

        if len(self.config.lang) == 1:
            return self.config.lang[0]

        lang_sample_start = max(0, timestamp - self.state.window_ts)
        lang_samples_size = min(2 * 16000, self.state.audio_ts - self.state.window_ts - lang_sample_start)

        lang, _ = self._guess_language(
            audio=self.state.rolling_window.array,
            sample_offset=lang_sample_start,
            sample_duration=lang_samples_size,
            lang = self.config.lang
            )
        return lang

    def transcribe_window(self) -> Tuple[List[VerbatimWord], List[VerbatimWord]]:
        LOG.info("Starting transcription of audio chunk.")
        LOG.info(f"Window Start Time: {samples_to_seconds(self.state.window_ts)}")
        LOG.info(f"Confirmed Time: {samples_to_seconds(self.state.confirmed_ts)}")
        LOG.info(f"Acknowledged Time: {samples_to_seconds(self.state.acknowledged_ts)}")
        LOG.info(f"Valid audio range: 0.0 - {samples_to_seconds(self.state.audio_ts - self.state.window_ts)}")

        acknowledged_words_in_window = WhisperHistory.advance_transcript(timestamp=self.state.window_ts, transcript=self.state.acknowledged_words)
        prefix_text = ''.join([w.word for u in self.state.unacknowledged_utterances for w in u.words])
        lang = self.guess_language(timestamp=max(0, self.state.acknowledged_ts))
        whisper_prompt = self.config.whisper_prompts[lang] if lang in self.config.whisper_prompts else self.config.whisper_prompts["en"]
        transcript_words = self.models.transcriber.transcribe(
            audio=self.state.rolling_window.array,
            lang=lang, prompt=whisper_prompt, prefix=prefix_text,
            window_ts=self.state.window_ts, audio_ts=self.state.audio_ts,
            whisper_beam_size = self.config.whisper_beam_size,
            whisper_best_of = self.config.whisper_best_of,
            whisper_patience = self.config.whisper_patience,
            whisper_temperatures = self.config.whisper_temperatures,
            )

        self.state.transcript_candidate_history.advance(self.state.window_ts)
        confirmed_words = self.state.transcript_candidate_history.confirm(
            current_words=transcript_words, after_ts=self.state.acknowledged_ts-1, prefix=acknowledged_words_in_window)
        self.state.transcript_candidate_history.add(transcription=transcript_words)

        if len(confirmed_words) > 0:
            self.state.confirmed_ts = confirmed_words[-1].start_ts
            LOG.debug(f"Confirmed ts: {self.state.confirmed_ts} ({samples_to_seconds(self.state.confirmed_ts)})")

            newline = os.linesep
            transcript_history = self.state.transcript_candidate_history.transcript_history
            LOG.debug(f"Transcript:\n{newline.join([''.join([w.word for w in history]) for history in transcript_history])}\n"
                        f"{''.join(w.word for w in transcript_words)}\n{''.join(w.word for w in confirmed_words)}")

        unconfirmed_words = [transcript_words[i] for i in range(len(confirmed_words), len(transcript_words))]
        return confirmed_words, unconfirmed_words

    def get_next_number_of_chunks(self):
        available_chunks = self.config.window_duration - float(self.state.audio_ts - self.state.window_ts) / self.config.sampling_rate

        thresholds = self.config.chunk_table

        for limit, value in thresholds:
            limit_sample = limit * self.config.window_duration
            value_sample = value * self.config.window_duration
            if available_chunks >= limit_sample:
                return value_sample

        # If for some reason available_chunks is less than 0, return the smallest chunk count.
        return 1

    def pretty_print_transcript(
        self,
        acknowledged_utterances:List[VerbatimUtterance],
        unacknowledged_utterances:List[VerbatimUtterance],
        unconfirmed_words:List[VerbatimWord],
        file:TextIO = sys.stdout
    ):
        formatter:TranscriptFormatter = TranscriptFormatter()
        file.write(
            f"[{samples_to_seconds(self.state.window_ts)}/"
            f"{samples_to_seconds(self.state.audio_ts - self.state.acknowledged_ts)}/"
            f"{samples_to_seconds(self.state.audio_ts - self.state.confirmed_ts)}]" + Fore.LIGHTGREEN_EX)
        for u in acknowledged_utterances:
            formatter.format_utterance(utterance=u, out=file, colours=COLORSCHEME_ACKNOWLEDGED)
        for u in unacknowledged_utterances:
            formatter.format_utterance(utterance=u, out=file, colours=COLORSCHEME_UNACKNOWLEDGED)
        if len(unconfirmed_words) > 0:
            formatter.format_utterance(utterance=VerbatimUtterance.from_words(unconfirmed_words), out=file, colours=COLORSCHEME_UNCONFIRMED)
        file.write(os.linesep)
        file.flush()

    def acknowledge_utterances(
            self,
            utterances:List[VerbatimUtterance],
            min_ack_duration=16000, min_unack_duration=16000,
    ) -> Tuple[List[VerbatimUtterance], List[VerbatimUtterance]]:
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

    def get_speaker_at(self, time:float, diarization:Annotation):
        if diarization is None:
            return None

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.end < time:
                continue
            if turn.start > time:
                break
            return speaker
        return None

    def get_speaker_before(self, time:float, diarization:Annotation):
        last_speaker = None
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.end < time:
                last_speaker = speaker
                continue
            if turn.start > time:
                break
        return last_speaker

    def get_speaker_after(self, time:float, diarization:Annotation):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start > time:
                return speaker
        return None

    def assign_speaker(self, utterance:VerbatimUtterance, diarization:Annotation):
        if diarization is None:
            return None

        start = utterance.start_ts / 16000.0
        end =utterance.end_ts / 16000.0
        duration = end - start
        samples:List[float]
        if duration < 1:
            samples = [ start + duration * 0.5]
        elif duration < 4:
            samples = [ start + duration * 0.25,
                        start + duration * 0.5,
                        start + duration * 0.75 ]
        else:
            samples = [ start + duration * 0.2,
                        start + duration * 0.3,
                        start + duration * 0.4,
                        start + duration * 0.5,
                        start + duration * 0.6,
                        start + duration * 0.7,
                        start + duration * 0.8 ]

        votes = {}
        for sample in samples:
            speaker = self.get_speaker_at(time=sample, diarization=diarization)
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

    def process_audio_window(self) -> Generator[Tuple[VerbatimUtterance,List[VerbatimUtterance],List[VerbatimWord]], None, None]:
        while True:
            # minimum number of samples to attempt transcription
            min_audio_duration_samples = 16000
            min_speech_duration_ms = 500
            utterances = []
            enable_vad = True
            if enable_vad and self.state.skip_silences:
                self.state.skip_silences = False
                self.skip_leading_silence(min_speech_duration_ms=min_speech_duration_ms)
                if self.state.audio_ts - self.state.window_ts < min_audio_duration_samples:
                    # we skipped all available audio - keep skipping silences and do nothing else for now
                    self.state.skip_silences = True

            if self.state.audio_ts - self.state.window_ts < min_audio_duration_samples:
                return

            if self.config.debug:
                self.dump_window_to_file(filename=f"{self.config.working_prefix_no_ext}-debug_window.wav")  # Dump current window for debugging

            confirmed_words, unconfirmed_words = self.transcribe_window()
            self.state.unconfirmed_words = unconfirmed_words
            if len(confirmed_words) > 0:
                utterances = self.words_to_sentences(word_tokenizer=self.models.sentence_tokenizer, window_words=confirmed_words)
                acknowledged_utterances, confirmed_utterances = self.acknowledge_utterances(utterances=utterances)

                if self.config.diarization:
                    for acknowledged_utterance in acknowledged_utterances:
                        acknowledged_utterance.speaker = self.assign_speaker(acknowledged_utterance, self.config.diarization)
                #elif self.config.stream and self.config.diarize:
                #    for acknowledged_utterance in acknowledged_utterances:
                #        self.diarize_utterance(acknowledged_utterance)

                self.state.acknowledged_utterances += acknowledged_utterances
                self.state.unacknowledged_utterances = confirmed_utterances

                for i, utterance in enumerate(acknowledged_utterances):
                    yield utterance, acknowledged_utterances[i+1:] + confirmed_utterances, unconfirmed_words

                if len(acknowledged_utterances) > 0:
                    for u in acknowledged_utterances:
                        self.state.acknowledged_words += u.words
                    
                    # utterances are split at short pauses, advance a bit to avoid repeating the last word
                    # but not too much as to skip the first word of the next utterance
                    utterance_padding_ms = 100
                    utterance_padding_samples = utterance_padding_ms * 16000 // 1000
                    
                    self.state.acknowledged_ts = acknowledged_utterances[-1].end_ts + utterance_padding_samples
                    self.state.skip_silences = True
            else:
                acknowledged_utterances = []
                confirmed_utterances = []


            outstr = StringIO()
            self.pretty_print_transcript(
                acknowledged_utterances=[], unacknowledged_utterances=confirmed_utterances, unconfirmed_words=unconfirmed_words, file=outstr)
            LOG.info(outstr.getvalue())

            self.state.acknowledged_words = WhisperHistory.advance_transcript(
                timestamp=self.state.window_ts, transcript=self.state.acknowledged_words)

            if self.state.acknowledged_ts > self.state.window_ts:
                shift_amount = self.state.acknowledged_ts - self.state.window_ts
                self.state.advance_audio_window(shift_amount)
                self.state.skip_silences = True

            if len(utterances) <= 1:
                break

    def capture_audio(self, audio_source:AudioSource):
        if not audio_source.has_more():
            return False

        next_chunk = self.get_next_number_of_chunks()
        LOG.debug(f"capturing {next_chunk} chunks of audio")
        audio_array = audio_source.next_chunk(next_chunk)
        self.state.append_audio_to_window(audio_array)
        return True

    def transcribe(self) -> Generator[Tuple[VerbatimUtterance,List[VerbatimUtterance],List[VerbatimWord]], None, None]:
        self.state.rolling_window.reset()  # Initialize empty rolling window

        try:
            self.config.source_stream.open()
            LOG.info("Starting main loop for audio transcription.")
            while True:
                has_more_audio = self.capture_audio(audio_source=self.config.source_stream)
                had_utterances = False

                # capture any utterance that slipped out of the current window
                flushed_utterances = []
                while len(self.state.unacknowledged_utterances) > 0:
                    if self.state.unacknowledged_utterances[0].end_ts > self.state.window_ts:
                        break
                    utterance = self.state.unacknowledged_utterances.pop(0)
                    utterance.speaker = self.assign_speaker(utterance, self.config.diarization)
                    yield utterance, self.state.unacknowledged_utterances, self.state.unconfirmed_words

                if len(self.state.unacknowledged_utterances) > 0:
                    flushed_utterances_words = []
                    partial_utterance = self.state.unacknowledged_utterances[0]
                    while len(partial_utterance.words) > 0:
                        if partial_utterance.words[0].end_ts > self.state.window_ts:
                            break
                        flushed_word = partial_utterance.words.pop(0)
                        flushed_utterances_words.append(flushed_word)
                        partial_utterance.start_ts = partial_utterance.words[-1].start_ts
                        partial_utterance.text = [w.word for w in partial_utterance.words]

                    if len(flushed_utterances_words) > 0:
                        utterance = VerbatimUtterance.from_words(flushed_utterances_words)
                        utterance.speaker = self.assign_speaker(utterance, self.config.diarization)
                        yield utterance, self.state.unacknowledged_utterances, self.state.unconfirmed_words
                else:
                    flushed_utterances_words = []
                    while len(self.state.unconfirmed_words) > 0:
                        if self.state.unconfirmed_words[0].end_ts > self.state.window_ts:
                            break
                        flushed_word = self.state.unconfirmed_words.pop(0)
                        flushed_utterances_words.append(flushed_word)

                    if len(flushed_utterances_words) > 0:
                        utterance = VerbatimUtterance.from_words(flushed_utterances_words)
                        utterance.speaker = self.assign_speaker(utterance, self.config.diarization)
                        yield utterance, self.state.unacknowledged_utterances, self.state.unconfirmed_words

                if len(flushed_utterances_words) > 0:
                    flushed_utterances.append(VerbatimUtterance.from_words(flushed_utterances_words))

                for utterance, unacknowmedged, unconfirmed in self.process_audio_window():
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
            LOG.info("Cleaning up resources.")

            for i, utterance in enumerate(self.state.unacknowledged_utterances):
                utterance.speaker = self.assign_speaker(utterance, self.config.diarization)
                yield utterance, self.state.unacknowledged_utterances[i+1:], self.state.unconfirmed_words

            if len(self.state.unconfirmed_words) > 0:
                unconfirmed_utterance:VerbatimUtterance = VerbatimUtterance.from_words(self.state.unconfirmed_words)
                unconfirmed_utterance.speaker = self.assign_speaker(unconfirmed_utterance, self.config.diarization)
                yield unconfirmed_utterance, [], []

            self.config.source_stream.close()

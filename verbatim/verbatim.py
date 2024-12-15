import logging
import os
import sys
import wave
from dataclasses import dataclass, field
from io import StringIO
from typing import List, Tuple, TextIO

import numpy as np
from colorama import Fore
from pyannote.core.annotation import Annotation

from .audio.sources.audiosource import AudioSource
from .transcript.sentences import FastSentenceTokenizer, SentenceTokenizer, SaTSentenceTokenizer
from .transcript.words import VerbatimWord, VerbatimUtterance
from .audio.audio import samples_to_seconds
from .config import Config
from .transcript.format.txt import TranscriptFormatter, COLORSCHEME_ACKNOWLEDGED, COLORSCHEME_UNACKNOWLEDGED, \
    COLORSCHEME_UNCONFIRMED
from .voices.silences import SileroVoiceActivityDetection, VoiceActivityDetection
from .voices.transcribe import WhisperTranscriber, Transcriber, APPEND_PUNCTUATIONS

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
        return [w for w in transcript if w.end_ts >= timestamp]

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
            LOG.info(f"CONFIRMED: {''.join([w.word for w in confirmed_words])}")
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
    skip_silences: bool = False
    speaker_embeddings:List = None


    def __init__(self):
        self.config = Config()
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
            self.advance_audio_window(shift_amount)
            self.skip_silences = True
            self.rolling_window.array[-chunk_size:] = audio_chunk

        self.audio_ts += chunk_size


class Verbatim:
    state:State = None
    config:Config = None

    def __init__(self, config:Config = Config()):
        self.state = State()
        self.config = config

        LOG.info("Initializing WhisperModel and audio stream.")
        self.transcriber:Transcriber = WhisperTranscriber(
            model_size_or_path=config.whisper_model_size, device=config.device,
            whisper_beam_size = config.whisper_beam_size,
            whisper_best_of = config.whisper_best_of,
            whisper_patience = config.whisper_patience,
            whisper_temperatures = config.whisper_temperatures
        )

        LOG.info("Initializing Silero VAD model.")
        self.vad:VoiceActivityDetection = SileroVoiceActivityDetection()

        LOG.info("Initializing Sentence Tokenizer.")
        if config.stream:
            self.sentence_tokenizer: SentenceTokenizer = FastSentenceTokenizer()
        else:
            self.sentence_tokenizer: SentenceTokenizer = SaTSentenceTokenizer(config.device)

    def skip_leading_silence(self) -> int:
        voice_segments = self.vad.find_activity(audio=self.state.rolling_window.array)
        LOG.debug(f"Voice segments: {voice_segments}")
        if len(voice_segments) == 0:
            return 0

        voice_start = voice_segments[0]['start']
        voice_end = voice_segments[0]['end']

        # skip leading silences
        if voice_start > 0:
            self.state.advance_audio_window(voice_start)
            voice_end -= voice_start
            return voice_end + self.state.window_ts

        # skip short audio sequences followed by a silence
        if voice_end - voice_start < 16000 and len(voice_segments) > 1:
            next_voice_start = voice_segments[1]['start']
            next_voice_end = voice_segments[1]['end']
            LOG.debug(f"Shifting rolling window by {next_voice_start} samples.")
            self.state.advance_audio_window(next_voice_start)
            next_voice_end -= next_voice_start
            return next_voice_end + self.state.window_ts

        return voice_end + self.state.window_ts

    def dump_window_to_file(self, filename: str = "debug_window.wav"):
        window = self.state.rolling_window.array
        LOG.debug("Dumping current window to file for debugging.")
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

        # For each sentence, we want to collect a sublist of VerbatimWord
        for sentence in sentences:
            sentence_length = len(sentence.strip())
            target_end = current_char_index + sentence_length

            sentence_words = []
            # Accumulate words until we have accounted for all chars of this sentence
            while current_char_index < target_end:
                # Pick the next word
                w = window_words[current_word_index]
                sentence_words.append(w)
                word_text = w.word
                if current_char_index == target_end - sentence_length:
                    word_text = word_text.lstrip()
                if current_char_index + len(word_text) > target_end:
                    word_text = word_text.rstrip()
                current_char_index += len(word_text)
                current_word_index += 1

                # If we overshoot, something is wrong,
                # but given the perfect alignment we expect to land exactly on target_end
                if current_char_index > target_end:
                    raise ValueError("Mismatch in alignment between sentences and words.")

            # Now we have all the words for this sentence
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
                 "to {sample_offset + sample_duration}({samples_to_seconds(sample_offset + sample_duration)})")
        return self.transcriber.guess_language(audio=lang_samples, lang=lang)


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
        prefix_text = ''.join([w.word for w in acknowledged_words_in_window])
        lang = self.guess_language(timestamp=max(0, self.state.acknowledged_ts))
        whisper_prompt = self.config.whisper_prompts[lang] if lang in self.config.whisper_prompts else self.config.whisper_prompts["en"]
        transcript_words = self.transcriber.transcribe(
            audio=self.state.rolling_window.array,
            lang=lang, prompt=whisper_prompt, prefix=prefix_text,
            window_ts=self.state.window_ts, audio_ts=self.state.audio_ts)

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
        available_chunks = (self.config.window_duration - float(self.state.audio_ts - self.state.window_ts) / self.config.sampling_rate)

        thresholds = self.config.chunk_table

        for limit, value in thresholds:
            if available_chunks >= limit:
                return value

        # If for some reason available_chunks is less than 0, return the smallest chunk count.
        return 1

    def pretty_print_transcript(self, acknowledged_utterances:List[VerbatimUtterance], unacknowledged_utterances:List[VerbatimUtterance], unconfirmed_words:List[VerbatimWord], file:TextIO = sys.stdout):
        formatter:TranscriptFormatter = TranscriptFormatter()
        file.write(f"[{samples_to_seconds(self.state.window_ts)}/{samples_to_seconds(self.state.audio_ts - self.state.acknowledged_ts)}/{samples_to_seconds(self.state.audio_ts - self.state.confirmed_ts)}]" + Fore.LIGHTGREEN_EX)
        for u in acknowledged_utterances:
            formatter.format_utterance(utterance=u, out=file, colours=COLORSCHEME_ACKNOWLEDGED)
        for u in unacknowledged_utterances:
            formatter.format_utterance(utterance=u, out=file, colours=COLORSCHEME_UNACKNOWLEDGED)
        if len(unconfirmed_words) > 0:
            formatter.format_utterance(utterance=VerbatimUtterance.from_words(unconfirmed_words), out=file, colours=COLORSCHEME_UNCONFIRMED)
        file.write(os.linesep)
        file.flush()

    def acknowledge_utterances(self, utterances:List[VerbatimUtterance]) -> Tuple[List[VerbatimUtterance], List[VerbatimUtterance]]:
        if len(utterances) == 0:
            return [], []
        if len(utterances) == 1:
            valid_endings = tuple(APPEND_PUNCTUATIONS)
            if utterances[0].text.endswith(valid_endings):
                return utterances, []
            else:
                return [], utterances
        return utterances[0:1], utterances[1:]

    def get_speaker_at(self, time:float, diarization:Annotation):
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

    def process_audio_window(self) -> List[VerbatimUtterance]:
        while True:
            utterances = []
            enable_vad = True
            if enable_vad and self.state.skip_silences:
                self.state.skip_silences = False

                voice_end = self.skip_leading_silence()

                if voice_end <= 0:
                    if self.state.audio_ts > self.state.window_ts:
                        remaining_audio = self.state.audio_ts - self.state.window_ts
                        self.state.advance_audio_window(remaining_audio)
                        self.state.skip_silences = True
                    return

            if self.config.debug:
                self.dump_window_to_file()  # Dump current window for debugging

            confirmed_words, unconfirmed_words = self.transcribe_window()
            self.state.unconfirmed_words = unconfirmed_words
            if len(confirmed_words) > 0:
                utterances = self.words_to_sentences(word_tokenizer=self.sentence_tokenizer, window_words=confirmed_words)
                acknowledged_utterances, confirmed_utterances = self.acknowledge_utterances(utterances=utterances)

                if self.config.diarization:
                    for acknowledged_utterance in acknowledged_utterances:
                        acknowledged_utterance.speaker = self.assign_speaker(acknowledged_utterance, self.config.diarization)
                #elif self.config.stream and self.config.diarize:
                #    for acknowledged_utterance in acknowledged_utterances:
                #        self.diarize_utterance(acknowledged_utterance)

                yield from acknowledged_utterances

                self.state.acknowledged_utterances += acknowledged_utterances
                self.state.unacknowledged_utterances = confirmed_utterances

                if len(acknowledged_utterances) > 0:
                    for u in acknowledged_utterances:
                        self.state.acknowledged_words += u.words
                    self.state.acknowledged_ts = acknowledged_utterances[-1].end_ts
            else:
                acknowledged_utterances = []
                confirmed_utterances = []


            outstr = StringIO()
            self.pretty_print_transcript(acknowledged_utterances=[], unacknowledged_utterances=confirmed_utterances, unconfirmed_words=unconfirmed_words, file=outstr)
            LOG.info(outstr.getvalue())

            self.state.acknowledged_words = WhisperHistory.advance_transcript(timestamp=self.state.window_ts, transcript=self.state.acknowledged_words)

            if self.state.acknowledged_ts > self.state.window_ts:
                shift_amount = self.state.acknowledged_ts - self.state.window_ts
                self.state.advance_audio_window(shift_amount)
                self.state.skip_silences = True

            if len(utterances) > 1:
                continue
            else:
                break

    def capture_audio(self, audio_source:AudioSource):
        if not audio_source.has_more():
            return False

        next_chunk = self.get_next_number_of_chunks()
        LOG.debug(f"capturing {next_chunk} chunks of audio")
        audio_array = audio_source.next_chunk(next_chunk)
        self.state.append_audio_to_window(audio_array)
        return True

    def transcribe(self) -> List[VerbatimUtterance]:
        self.state.rolling_window.reset()  # Initialize empty rolling window

        try:
            self.config.source.open()
            LOG.info("Starting main loop for audio transcription.")
            while True:
                has_more_audio = self.capture_audio(audio_source=self.config.source)
                had_utterances = False

                for u in self.process_audio_window():
                    had_utterances = True
                    yield u

                if not had_utterances and not has_more_audio:
                    break

        except KeyboardInterrupt:
            LOG.info("KeyboardInterrupt detected, stopping transcription.")
            LOG.debug("Stopping...")
        except Exception as e:
            import traceback
            LOG.error(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
            LOG.debug("Stopping...")
        finally:
            LOG.info("Cleaning up resources.")

            yield from self.state.unacknowledged_utterances

            if len(self.state.unconfirmed_words) > 0:
                unconfirmed_utterance:VerbatimUtterance = VerbatimUtterance.from_words(self.state.unconfirmed_words)
                yield unconfirmed_utterance

            self.config.source.close()

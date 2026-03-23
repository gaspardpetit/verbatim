import unittest

import numpy as np

from verbatim.config import Config
from verbatim.transcript.words import Word
from verbatim.verbatim import Verbatim
from verbatim_audio.sources.audiosource import AudioStream


class DummyAudioStream(AudioStream):
    def __init__(self):
        super().__init__(start_offset=0, diarization=None)

    def has_more(self) -> bool:
        return False

    def next_chunk(self, chunk_length=1):
        _ = chunk_length
        return np.zeros(0, dtype=np.float32)

    def close(self):
        return None

    def get_nchannels(self) -> int:
        return 1

    def get_rate(self) -> int:
        return 16000


class DummySentenceTokenizer:
    def split(self, words):
        return [word.word for word in words]


class DummyTranscriber:
    def __init__(self, outputs_by_lang):
        self.outputs_by_lang = outputs_by_lang
        self.calls = []

    def transcribe(
        self,
        *,
        audio,
        lang,
        prompt,
        prefix,
        window_ts,
        audio_ts,
        whisper_beam_size,
        whisper_best_of,
        whisper_patience,
        whisper_temperatures,
    ):
        _ = audio, prompt, prefix, whisper_beam_size, whisper_best_of, whisper_patience, whisper_temperatures
        self.calls.append(
            {
                "lang": lang,
                "window_ts": window_ts,
                "audio_ts": audio_ts,
            }
        )
        return list(self.outputs_by_lang.get(lang, []))


class DummyModels:
    def __init__(self, *, transcriber, sentence_tokenizer):
        self.transcriber = transcriber
        self.sentence_tokenizer = sentence_tokenizer


class LanguageAwareVerbatim(Verbatim):
    def __init__(self, *, config, models, guess_language_fn):
        super().__init__(config=config, models=models)
        self._guess_language_fn = guess_language_fn

    def guess_language(self, timestamp):
        return self._guess_language_fn(timestamp)


def make_word(start_ts: int, end_ts: int, text: str, lang: str = "en") -> Word:
    return Word(start_ts=start_ts, end_ts=end_ts, word=text, probability=1.0, lang=lang)


class TestInterUtteranceLanguageSwitch(unittest.TestCase):
    def test_process_audio_window_acknowledges_initial_utterance_when_only_switched_tail_is_missing(self):
        first_fr = make_word(0, 8000, " bonjour.", lang="fr")
        transcriber = DummyTranscriber(outputs_by_lang={"fr": [first_fr], "en": []})

        def guess_language_fn(timestamp):
            if timestamp == 0:
                return ("fr", 1.0, 16000)
            return ("en", 1.0, 16000)

        config = Config(device="cpu")
        config.lang = ["fr", "en"]
        verbatim = LanguageAwareVerbatim(
            config=config,
            models=DummyModels(
                transcriber=transcriber,
                sentence_tokenizer=DummySentenceTokenizer(),
            ),
            guess_language_fn=guess_language_fn,
        )
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = 50000
        verbatim.state.skip_silences = False
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        # Audio shape being modeled:
        #   FR1 EN2 EN3 EN4 EN5
        # Current pass is anchored in French and the ASR only returns FR1.
        #
        # This is intentionally not a red test. It documents the current behavior so we do not
        # overgeneralize the earlier regression family: in this configuration the pipeline still
        # acknowledges FR1, because `acknowledge_utterances()` does not require later same-language
        # content once the utterance is complete and within the duration bounds.
        verbatim.state.transcript_candidate_history.add([first_fr])

        emitted = list(verbatim.process_audio_window(audio_stream=DummyAudioStream()))

        self.assertEqual([" bonjour."], [utterance.text for utterance, _, _ in emitted])
        self.assertEqual("fr", transcriber.calls[0]["lang"])
        self.assertGreater(verbatim.state.window_ts, first_fr.end_ts)

    def test_process_audio_window_can_skip_missing_switched_utterance(self):
        first_en = make_word(0, 8000, " hello.", lang="en")
        second_en = make_word(22000, 30000, " welcome.", lang="en")
        transcriber = DummyTranscriber(outputs_by_lang={"en": [first_en, second_en], "fr": []})

        def guess_language_fn(timestamp):
            if timestamp == 0:
                return ("en", 1.0, 16000)
            if 8000 < timestamp < 22000:
                return ("fr", 1.0, 16000)
            return ("en", 1.0, 16000)

        config = Config(device="cpu")
        config.lang = ["en", "fr"]
        verbatim = LanguageAwareVerbatim(
            config=config,
            models=DummyModels(
                transcriber=transcriber,
                sentence_tokenizer=DummySentenceTokenizer(),
            ),
            guess_language_fn=guess_language_fn,
        )
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = 50000
        verbatim.state.skip_silences = False
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        # Seed confirmation history so the first pass can acknowledge both English utterances.
        # This simulates the case where the ASR, forced to English for the whole window, never
        # surfaced the switched French utterance that actually started between them.
        verbatim.state.transcript_candidate_history.add([first_en, second_en])

        emitted = list(verbatim.process_audio_window(audio_stream=DummyAudioStream()))

        # Desired behavior for the upcoming refactor:
        # once language would flip between acknowledged utterances, the pass should stop before
        # accepting the second English utterance. Current code acknowledges both and advances
        # past the missing French region, so this assertion fails today.
        self.assertEqual([" hello."], [utterance.text for utterance, _, _ in emitted])
        self.assertLess(verbatim.state.window_ts, second_en.start_ts)

    def test_process_audio_window_can_skip_missing_switched_utterance_after_overflow(self):
        missing_en = make_word(9000, 15000, " hello.", lang="en")
        first_fr_after_gap = make_word(22000, 30000, " bonjour.", lang="fr")
        trailing_fr_partial = make_word(31000, 36000, " salut", lang="fr")
        transcriber = DummyTranscriber(outputs_by_lang={"fr": [first_fr_after_gap, trailing_fr_partial], "en": []})

        def guess_language_fn(timestamp):
            if timestamp == 7000:
                return ("fr", 1.0, 16000)
            if missing_en.start_ts <= timestamp <= missing_en.end_ts:
                return ("en", 1.0, 16000)
            return ("fr", 1.0, 16000)

        config = Config(device="cpu")
        config.lang = ["fr", "en"]
        verbatim = LanguageAwareVerbatim(
            config=config,
            models=DummyModels(
                transcriber=transcriber,
                sentence_tokenizer=DummySentenceTokenizer(),
            ),
            guess_language_fn=guess_language_fn,
        )
        verbatim.state.window_ts = 7000
        verbatim.state.acknowledged_ts = 7000
        verbatim.state.audio_ts = 50000
        verbatim.state.skip_silences = False
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        # Simulate a prior overflow/advance that already moved us near the end of FR1.
        # The next French-only transcription pass surfaces FR3 and FR4, but EN2 remains missing.
        verbatim.state.transcript_candidate_history.add([first_fr_after_gap, trailing_fr_partial])

        emitted = list(verbatim.process_audio_window(audio_stream=DummyAudioStream()))

        # Desired behavior for the upcoming refactor:
        # do not acknowledge content that starts after an undetected language-switched gap.
        # Current code acknowledges FR3 and advances to FR4.start_ts, which is already past EN2.
        self.assertEqual([], [utterance.text for utterance, _, _ in emitted])
        self.assertLessEqual(verbatim.state.window_ts, missing_en.start_ts)

    def test_process_audio_window_can_advance_past_missing_switched_gap_with_incomplete_tail(self):
        first_fr = make_word(0, 8000, " bonjour.", lang="fr")
        missing_en = make_word(9000, 15000, " hello.", lang="en")
        trailing_fr_partial = make_word(22000, 26000, " salut", lang="fr")
        transcriber = DummyTranscriber(outputs_by_lang={"fr": [first_fr, trailing_fr_partial], "en": []})

        def guess_language_fn(timestamp):
            if timestamp == 0:
                return ("fr", 1.0, 16000)
            if missing_en.start_ts <= timestamp <= missing_en.end_ts:
                return ("en", 1.0, 16000)
            return ("fr", 1.0, 16000)

        config = Config(device="cpu")
        config.lang = ["fr", "en"]
        verbatim = LanguageAwareVerbatim(
            config=config,
            models=DummyModels(
                transcriber=transcriber,
                sentence_tokenizer=DummySentenceTokenizer(),
            ),
            guess_language_fn=guess_language_fn,
        )
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = 50000
        verbatim.state.skip_silences = False
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        # The French ASR sees FR1 plus an incomplete FR3 tail, but still misses EN2 in the middle.
        # Current logic acknowledges FR1 and advances into the missing EN2 span because the pending
        # blocker starts at FR3, not at the missing switched utterance.
        verbatim.state.transcript_candidate_history.add([first_fr, trailing_fr_partial])

        emitted = list(verbatim.process_audio_window(audio_stream=DummyAudioStream()))

        self.assertEqual([" bonjour."], [utterance.text for utterance, _, _ in emitted])
        self.assertLessEqual(verbatim.state.window_ts, missing_en.start_ts)


if __name__ == "__main__":
    unittest.main()

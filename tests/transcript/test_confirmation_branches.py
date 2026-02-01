import unittest

import numpy as np

from verbatim.config import Config
from verbatim.transcript.words import Utterance, Word
from verbatim.verbatim import Verbatim, WhisperHistory


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
        self.calls.append(
            {
                "lang": lang,
                "prefix": prefix,
                "window_ts": window_ts,
                "audio_ts": audio_ts,
            }
        )
        _ = audio, prompt, whisper_beam_size, whisper_best_of, whisper_patience, whisper_temperatures
        return list(self.outputs_by_lang.get(lang, []))


class DummyModels:
    def __init__(self, transcriber):
        self.transcriber = transcriber


class TestVerbatim(Verbatim):
    def __init__(self, config, models, guess_language_fn):
        super().__init__(config=config, models=models)
        self._guess_language_fn = guess_language_fn

    def guess_language(self, timestamp):
        return self._guess_language_fn(timestamp)


def make_word(start_ts, end_ts, text, lang="en"):
    return Word(start_ts=start_ts, end_ts=end_ts, word=text, probability=1.0, lang=lang)


class TestConfirmationBranches(unittest.TestCase):
    def setUp(self):
        self.config = Config(device="cpu")

    def test_prefix_text_stops_on_language_change(self):
        transcriber = DummyTranscriber(outputs_by_lang={"en": []})

        def guess_language_fn(_timestamp):
            return ("en", 1.0, 0)

        verbatim = TestVerbatim(config=self.config, models=DummyModels(transcriber), guess_language_fn=guess_language_fn)
        verbatim.state.audio_ts = 16000
        verbatim.state.window_ts = 0
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        words = [
            make_word(0, 10, " hello", lang="en"),
            make_word(10, 20, " welt", lang="de"),
            make_word(20, 30, " again", lang="en"),
        ]
        verbatim.state.unacknowledged_utterances = [Utterance.from_words("utt0", words)]

        verbatim.transcribe_window()

        self.assertEqual(1, len(transcriber.calls))
        self.assertEqual(" hello", transcriber.calls[0]["prefix"])

    def test_language_switch_selects_alternative(self):
        en_words = [make_word(12000, 13000, " hi", lang="en")]
        de_words = [make_word(9000, 10000, " hallo", lang="de")]
        transcriber = DummyTranscriber(outputs_by_lang={"en": en_words, "de": de_words})

        def guess_language_fn(timestamp):
            if timestamp == 0:
                return ("en", 1.0, 8000)
            if timestamp == 12000:
                return ("de", 1.0, 8000)
            if timestamp == 9000:
                return ("de", 1.0, 8000)
            return ("en", 1.0, 8000)

        config = Config(device="cpu")
        config.lang = ["en", "de"]
        verbatim = TestVerbatim(config=config, models=DummyModels(transcriber), guess_language_fn=guess_language_fn)
        verbatim.state.audio_ts = 20000
        verbatim.state.window_ts = 0
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        confirmed, unconfirmed = verbatim.transcribe_window()

        self.assertEqual([], confirmed)
        self.assertEqual(de_words, unconfirmed)
        self.assertEqual(["en", "de"], [call["lang"] for call in transcriber.calls])

    def test_language_switch_skipped_when_first_word_before_used_samples(self):
        en_words = [make_word(1000, 2000, " hi", lang="en")]
        transcriber = DummyTranscriber(outputs_by_lang={"en": en_words, "de": []})

        def guess_language_fn(_timestamp):
            return ("en", 1.0, 8000)

        config = Config(device="cpu")
        config.lang = ["en", "de"]
        verbatim = TestVerbatim(config=config, models=DummyModels(transcriber), guess_language_fn=guess_language_fn)
        verbatim.state.audio_ts = 20000
        verbatim.state.window_ts = 0
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)

        verbatim.transcribe_window()

        self.assertEqual(1, len(transcriber.calls))
        self.assertEqual("en", transcriber.calls[0]["lang"])

    def test_confirm_transcript_stops_on_mismatch(self):
        w1 = make_word(0, 5, " a")
        w2 = make_word(10, 15, " b")
        w3 = make_word(20, 25, " c")
        mismatch = make_word(10, 15, " x")

        confirmed = WhisperHistory.confirm_transcript(
            current_words=[w1, w2, w3],
            transcript=[w1, mismatch, w3],
            prefix=[],
            after_ts=-1,
        )

        self.assertEqual([w1], confirmed)

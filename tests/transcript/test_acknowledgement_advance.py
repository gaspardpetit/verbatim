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
    def __init__(self, sentences):
        self._sentences = sentences

    def split(self, words):
        _ = words
        return list(self._sentences)


class DummyModels:
    def __init__(self, sentence_tokenizer):
        self.sentence_tokenizer = sentence_tokenizer


class StubVerbatim(Verbatim):
    def __init__(self, *, config, models, confirmed_words, unconfirmed_words):
        super().__init__(config=config, models=models)
        self._confirmed_words = confirmed_words
        self._unconfirmed_words = unconfirmed_words

    def transcribe_window(self):
        return list(self._confirmed_words), list(self._unconfirmed_words)


def make_word(start_ts: int, end_ts: int, text: str) -> Word:
    return Word(start_ts=start_ts, end_ts=end_ts, word=text, probability=1.0, lang="en")


class TestAcknowledgementAdvance(unittest.TestCase):
    def setUp(self):
        self.audio_stream = DummyAudioStream()

    def _make_verbatim(self, *, sentences, confirmed_words, unconfirmed_words, audio_ts=50000):
        config = Config(device="cpu")
        verbatim = StubVerbatim(
            config=config,
            models=DummyModels(sentence_tokenizer=DummySentenceTokenizer(sentences)),
            confirmed_words=confirmed_words,
            unconfirmed_words=unconfirmed_words,
        )
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = audio_ts
        verbatim.state.skip_silences = False
        verbatim.state.rolling_window.array = np.zeros(len(verbatim.state.rolling_window.array), dtype=np.float32)
        return verbatim

    def test_advance_clamps_to_next_unacknowledged_utterance(self):
        hello = make_word(0, 16000, " hello.")
        next_word = make_word(17000, 33000, " next.")
        verbatim = self._make_verbatim(
            sentences=[" hello.", " next."],
            confirmed_words=[hello, next_word],
            unconfirmed_words=[],
        )

        emitted = list(verbatim.process_audio_window(audio_stream=self.audio_stream))

        self.assertEqual(1, len(emitted))
        self.assertEqual(17000, verbatim.state.acknowledged_ts)
        self.assertEqual(17000, verbatim.state.window_ts)
        self.assertEqual(" next.", verbatim.state.unacknowledged_utterances[0].text)

    def test_advance_clamps_to_next_unconfirmed_word(self):
        hello = make_word(0, 16000, " hello.")
        next_word = make_word(16500, 20000, " next")
        verbatim = self._make_verbatim(
            sentences=[" hello."],
            confirmed_words=[hello],
            unconfirmed_words=[next_word],
        )

        emitted = list(verbatim.process_audio_window(audio_stream=self.audio_stream))

        self.assertEqual(1, len(emitted))
        self.assertEqual(16500, verbatim.state.acknowledged_ts)
        self.assertEqual(16500, verbatim.state.window_ts)
        self.assertEqual([next_word], verbatim.state.unconfirmed_words)


if __name__ == "__main__":
    unittest.main()

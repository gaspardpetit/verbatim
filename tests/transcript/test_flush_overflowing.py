import unittest

from verbatim.config import Config
from verbatim.transcript.words import Utterance, Word
from verbatim.verbatim import Verbatim


class DummyModels:
    pass


def make_word(start_ts: int, end_ts: int, text: str = " w") -> Word:
    return Word(start_ts=start_ts, end_ts=end_ts, word=text, probability=1.0, lang="en")


class TestFlushOverflowing(unittest.TestCase):
    def setUp(self):
        self.config = Config(device="cpu")
        self.verbatim = Verbatim(config=self.config, models=DummyModels())

    def test_flush_acknowledges_unconfirmed_words(self):
        w1 = make_word(0, 10, " a")
        w2 = make_word(10, 20, " b")
        w3 = make_word(60, 70, " c")
        self.verbatim.state.window_ts = 50
        self.verbatim.state.unconfirmed_words = [w1, w2, w3]

        flushed = list(self.verbatim.flush_overflowing_utterances(diarization=None))

        self.assertEqual(1, len(flushed))
        flushed_utterance, _, _ = flushed[0]
        self.assertEqual([w1, w2], flushed_utterance.words)
        self.assertEqual([w3], self.verbatim.state.unconfirmed_words)
        self.assertEqual([w1, w2], self.verbatim.state.acknowledged_words)

    def test_flush_partial_utterance_updates_remaining_start(self):
        w1 = make_word(0, 10, " a")
        w2 = make_word(40, 60, " b")
        utterance = Utterance.from_words(utterance_id="utt0", words=[w1, w2])
        self.verbatim.state.window_ts = 50
        self.verbatim.state.unacknowledged_utterances = [utterance]

        flushed = list(self.verbatim.flush_overflowing_utterances(diarization=None))

        self.assertEqual(1, len(flushed))
        flushed_utterance, _, _ = flushed[0]
        self.assertEqual([w1], flushed_utterance.words)
        self.assertEqual([w1], self.verbatim.state.acknowledged_words)
        self.assertEqual(1, len(self.verbatim.state.unacknowledged_utterances))
        remaining = self.verbatim.state.unacknowledged_utterances[0]
        self.assertEqual(w2.start_ts, remaining.start_ts)
        self.assertEqual(w2.word, remaining.text)
        self.assertEqual([w2], remaining.words)

    def test_flush_partial_utterance_removes_empty(self):
        w1 = make_word(0, 10, " a")
        utterance = Utterance.from_words(utterance_id="utt0", words=[w1])
        self.verbatim.state.window_ts = 50
        self.verbatim.state.unacknowledged_utterances = [utterance]

        flushed = list(self.verbatim.flush_overflowing_utterances(diarization=None))

        self.assertEqual(1, len(flushed))
        self.assertEqual([], self.verbatim.state.unacknowledged_utterances)
        self.assertEqual([w1], self.verbatim.state.acknowledged_words)

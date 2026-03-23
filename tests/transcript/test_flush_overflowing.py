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

    def test_flush_crossing_utterance_acknowledges_whole_utterance(self):
        w1 = make_word(0, 10, " a")
        w2 = make_word(40, 60, " b")
        utterance = Utterance.from_words(utterance_id="utt0", words=[w1, w2])
        self.verbatim.state.window_ts = 50
        self.verbatim.state.unacknowledged_utterances = [utterance]

        flushed = list(self.verbatim.flush_overflowing_utterances(diarization=None))

        self.assertEqual(1, len(flushed))
        flushed_utterance, _, _ = flushed[0]
        self.assertEqual([w1, w2], flushed_utterance.words)
        self.assertEqual([w1, w2], self.verbatim.state.acknowledged_words)
        self.assertEqual([], self.verbatim.state.unacknowledged_utterances)

    def test_flush_crossing_single_word_utterance_removes_empty(self):
        w1 = make_word(0, 10, " a")
        utterance = Utterance.from_words(utterance_id="utt0", words=[w1])
        self.verbatim.state.window_ts = 50
        self.verbatim.state.unacknowledged_utterances = [utterance]

        flushed = list(self.verbatim.flush_overflowing_utterances(diarization=None))

        self.assertEqual(1, len(flushed))
        self.assertEqual([], self.verbatim.state.unacknowledged_utterances)
        self.assertEqual([w1], self.verbatim.state.acknowledged_words)

    def test_flush_prefers_whole_crossing_utterance_over_word_split(self):
        w1 = make_word(0, 10, " a")
        w2 = make_word(40, 60, " b")
        crossing = Utterance.from_words(utterance_id="utt0", words=[w1, w2])
        self.verbatim.state.window_ts = 50
        self.verbatim.state.unacknowledged_utterances = [crossing]

        flushed = list(self.verbatim.flush_overflowing_utterances(diarization=None))

        # Desired behavior for the refactor:
        # if an utterance crosses the window boundary, flush the whole utterance so the
        # remaining state starts at the next utterance boundary instead of mid-utterance.
        flushed_utterance, _, _ = flushed[0]
        self.assertEqual([w1, w2], flushed_utterance.words)
        self.assertEqual([], self.verbatim.state.unacknowledged_utterances)
        self.assertEqual([w1, w2], self.verbatim.state.acknowledged_words)

    def test_flush_prefers_next_utterance_boundary_when_one_utterance_crosses(self):
        w1 = make_word(0, 10, " a")
        w2 = make_word(40, 60, " b")
        w3 = make_word(70, 80, " c")
        crossing = Utterance.from_words(utterance_id="utt0", words=[w1, w2])
        following = Utterance.from_words(utterance_id="utt1", words=[w3])
        self.verbatim.state.window_ts = 50
        self.verbatim.state.unacknowledged_utterances = [crossing, following]

        flushed = list(self.verbatim.flush_overflowing_utterances(diarization=None))

        flushed_utterance, remaining_unack, _ = flushed[0]
        self.assertEqual([w1, w2], flushed_utterance.words)
        self.assertEqual([following], remaining_unack)
        self.assertEqual([following], self.verbatim.state.unacknowledged_utterances)
        self.assertEqual([w1, w2], self.verbatim.state.acknowledged_words)

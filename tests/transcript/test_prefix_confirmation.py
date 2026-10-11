import unittest

from verbatim.transcript.words import Word
from verbatim.verbatim import WhisperHistory


def make_word(start_ts: int, end_ts: int, text: str) -> Word:
    return Word(start_ts=start_ts, end_ts=end_ts, word=text, probability=1.0, lang="en")


class TestPrefixConfirmation(unittest.TestCase):
    def test_confirm_after_ts_inclusive(self):
        w1 = make_word(0, 5, " a")
        w2 = make_word(10, 15, " b")
        w3 = make_word(20, 25, " c")
        current_words = [w1, w2, w3]
        transcript = [w1, w2, w3]

        confirmed = WhisperHistory.confirm_transcript(
            current_words=current_words,
            transcript=transcript,
            prefix=[],
            after_ts=10,
        )

        self.assertEqual([w3], confirmed)

    def test_confirm_after_ts_exclusive_when_below_start(self):
        w1 = make_word(0, 5, " a")
        w2 = make_word(10, 15, " b")
        w3 = make_word(20, 25, " c")
        current_words = [w1, w2, w3]
        transcript = [w1, w2, w3]

        confirmed = WhisperHistory.confirm_transcript(
            current_words=current_words,
            transcript=transcript,
            prefix=[],
            after_ts=9,
        )

        self.assertEqual([w2, w3], confirmed)

    def test_confirm_skips_prefix_matches(self):
        w1 = make_word(0, 5, " a")
        w2 = make_word(10, 15, " b")
        w3 = make_word(20, 25, " c")
        current_words = [w1, w2, w3]
        transcript = [w1, w2, w3]
        prefix = [w1, w2]

        confirmed = WhisperHistory.confirm_transcript(
            current_words=current_words,
            transcript=transcript,
            prefix=prefix,
            after_ts=-1,
        )

        self.assertEqual([w3], confirmed)

    def test_advance_transcript_keeps_end_equal_timestamp(self):
        w1 = make_word(0, 10, " a")
        w2 = make_word(11, 20, " b")

        advanced = WhisperHistory.advance_transcript(timestamp=10, transcript=[w1, w2])

        self.assertEqual([w1, w2], advanced)

    def test_same_word_later_in_audio_is_not_skipped_as_prefix(self):
        acknowledged = make_word(0, 5, " you")
        repeated = make_word(20, 25, " you")

        self.assertEqual(0, WhisperHistory.skipped_prefix_length([repeated], [acknowledged], after_ts=5))

    def test_same_word_touching_acknowledged_boundary_is_not_skipped(self):
        acknowledged = make_word(0, 5, " you")
        repeated = make_word(5, 10, " you")

        self.assertEqual(0, WhisperHistory.skipped_prefix_length([repeated], [acknowledged], after_ts=4))
        self.assertEqual(
            [repeated],
            WhisperHistory.confirm_transcript([repeated], [repeated], [acknowledged], after_ts=4),
        )

    def test_zero_duration_acknowledged_word_at_same_timestamp_is_skipped(self):
        acknowledged = make_word(5, 5, " you")
        repeated = make_word(5, 5, " you")
        next_word = make_word(5, 10, " go")

        self.assertEqual(1, WhisperHistory.skipped_prefix_length([repeated, next_word], [acknowledged], after_ts=4))
        self.assertEqual(
            [next_word],
            WhisperHistory.confirm_transcript([repeated, next_word], [repeated, next_word], [acknowledged], after_ts=4),
        )

    def test_zero_duration_word_with_different_text_is_not_skipped(self):
        acknowledged = make_word(5, 5, " you")
        next_word = make_word(5, 5, " go")

        self.assertEqual(0, WhisperHistory.skipped_prefix_length([next_word], [acknowledged], after_ts=4))

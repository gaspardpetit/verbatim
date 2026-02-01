import unittest

from verbatim.transcript.idprovider import CounterIdProvider
from verbatim.transcript.words import Word
from verbatim.verbatim import Verbatim


class TestSentenceAlignment(unittest.TestCase):
    def test_align_words_handles_apostrophe_split(self):
        words = [
            Word(start_ts=0, end_ts=1, word=" and", probability=1.0, lang="en"),
            Word(start_ts=1, end_ts=2, word=" then", probability=1.0, lang="en"),
            Word(start_ts=2, end_ts=3, word=" i'll", probability=1.0, lang="en"),
            Word(start_ts=3, end_ts=4, word=" um", probability=1.0, lang="en"),
        ]

        sentences = [" and then i", "'ll um"]

        utterances = Verbatim.align_words_to_sentences(
            id_provider=CounterIdProvider(prefix="utt"),
            sentences=sentences,
            window_words=words,
        )

        # Expect two utterances whose concatenated text matches the input words.
        self.assertEqual(len(utterances), 2)
        self.assertEqual("".join(utterance.text for utterance in utterances), "".join(word.word for word in words))


if __name__ == "__main__":
    unittest.main()

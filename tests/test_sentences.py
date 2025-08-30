import logging
import pytest
import unittest

from verbatim.transcript.words import Word

# Configure logger
LOG = logging.getLogger(__name__)

WORDS_ALICE = [
    Word(start_ts=2135168, end_ts=2140608, word=" Alice", probability=1.0, lang="fr"),
    Word(start_ts=2140608, end_ts=2144448, word=" was", probability=1.0, lang="fr"),
    Word(start_ts=2145728, end_ts=2147648, word=" beginning", probability=1.0, lang="fr"),
    Word(start_ts=2149568, end_ts=2161088, word=" to", probability=1.0, lang="fr"),
    Word(start_ts=2161088, end_ts=2171328, word=" get", probability=1.0, lang="fr"),
    Word(start_ts=2171328, end_ts=2174528, word=" very", probability=1.0, lang="fr"),
    Word(start_ts=2175488, end_ts=2195328, word=" tired", probability=1.0, lang="fr"),
    Word(start_ts=2195328, end_ts=2200768, word=" of", probability=1.0, lang="fr"),
    Word(start_ts=2200768, end_ts=2202368, word=" sitting", probability=1.0, lang="fr"),
    Word(start_ts=2202368, end_ts=2203328, word=" by", probability=1.0, lang="fr"),
    Word(start_ts=2203328, end_ts=2206208, word=" her", probability=1.0, lang="fr"),
    Word(start_ts=2206208, end_ts=2208128, word=" sister", probability=1.0, lang="fr"),
    Word(start_ts=2208128, end_ts=2211968, word=" on", probability=1.0, lang="fr"),
    Word(start_ts=2211968, end_ts=2215808, word=" the", probability=1.0, lang="fr"),
    Word(start_ts=2215808, end_ts=2219328, word=" bank,", probability=1.0, lang="fr"),
    Word(start_ts=2219328, end_ts=2222528, word=" and", probability=1.0, lang="fr"),
    Word(start_ts=2222528, end_ts=2227328, word=" of", probability=1.0, lang="fr"),
    Word(start_ts=2227328, end_ts=2231488, word=" having", probability=1.0, lang="fr"),
    Word(start_ts=2231488, end_ts=2247488, word=" nothing", probability=1.0, lang="fr"),
    Word(start_ts=2247488, end_ts=2249728, word=" to", probability=1.0, lang="fr"),
    Word(start_ts=2249728, end_ts=2253248, word=" do:", probability=1.0, lang="fr"),
    Word(start_ts=2253248, end_ts=2254848, word=" once", probability=1.0, lang="fr"),
    Word(start_ts=2254848, end_ts=2258688, word=" or", probability=1.0, lang="fr"),
    Word(start_ts=2260608, end_ts=2262528, word=" twice", probability=1.0, lang="fr"),
    Word(start_ts=2262528, end_ts=2263807, word=" she", probability=1.0, lang="fr"),
    Word(start_ts=2263807, end_ts=2265408, word=" had", probability=1.0, lang="fr"),
    Word(start_ts=2265408, end_ts=2267008, word=" peeped", probability=1.0, lang="fr"),
    Word(start_ts=2267008, end_ts=2277888, word=" into", probability=1.0, lang="fr"),
    Word(start_ts=2278848, end_ts=2280448, word=" the", probability=1.0, lang="fr"),
    Word(start_ts=2280448, end_ts=2284288, word=" book", probability=1.0, lang="fr"),
    Word(start_ts=2284288, end_ts=2285568, word=" her", probability=1.0, lang="fr"),
    Word(start_ts=2285568, end_ts=2287808, word=" sister", probability=1.0, lang="fr"),
    Word(start_ts=2287808, end_ts=2294528, word=" was", probability=1.0, lang="fr"),
    Word(start_ts=2294528, end_ts=2298048, word=" reading,", probability=1.0, lang="fr"),
    Word(start_ts=2298048, end_ts=2301888, word=" but", probability=1.0, lang="fr"),
    Word(start_ts=2301888, end_ts=2306688, word=" it", probability=1.0, lang="fr"),
    Word(start_ts=2306688, end_ts=2320768, word=" had", probability=1.0, lang="fr"),
    Word(start_ts=2320768, end_ts=2323648, word=" no", probability=1.0, lang="fr"),
    Word(start_ts=2323648, end_ts=2335168, word=" pictures", probability=1.0, lang="fr"),
    Word(start_ts=2335168, end_ts=2350208, word=" or", probability=1.0, lang="fr"),
    Word(start_ts=2350208, end_ts=2357568, word=" conversations", probability=1.0, lang="fr"),
    Word(start_ts=2357568, end_ts=2360768, word=" in", probability=1.0, lang="fr"),
    Word(start_ts=2360768, end_ts=2362368, word=" it,", probability=1.0, lang="fr"),
    Word(start_ts=2362368, end_ts=2364608, word=" “and", probability=1.0, lang="fr"),
    Word(start_ts=2364608, end_ts=2365888, word=" what", probability=1.0, lang="fr"),
    Word(start_ts=2365888, end_ts=2369408, word=" is", probability=1.0, lang="fr"),
    Word(start_ts=2369408, end_ts=2376128, word=" the", probability=1.0, lang="fr"),
    Word(start_ts=2376128, end_ts=2378688, word=" use", probability=1.0, lang="fr"),
    Word(start_ts=2378688, end_ts=2383168, word=" of", probability=1.0, lang="fr"),
    Word(start_ts=2383168, end_ts=2384448, word=" a", probability=1.0, lang="fr"),
    Word(start_ts=2384448, end_ts=2385728, word=" book,”", probability=1.0, lang="fr"),
    Word(start_ts=2385728, end_ts=2386048, word=" thought", probability=1.0, lang="fr"),
    Word(start_ts=2386048, end_ts=2389248, word=" Alice", probability=1.0, lang="fr"),
    Word(start_ts=2389248, end_ts=2390848, word=" “without", probability=1.0, lang="fr"),
    Word(start_ts=2390848, end_ts=2393728, word=" pictures", probability=1.0, lang="fr"),
    Word(start_ts=2393728, end_ts=2395328, word=" or", probability=1.0, lang="fr"),
    Word(start_ts=2395328, end_ts=2397888, word=" conversations?”", probability=1.0, lang="fr"),
    Word(start_ts=2397888, end_ts=2407808, word=" So", probability=1.0, lang="fr"),
    Word(start_ts=2407808, end_ts=2417408, word=" she", probability=1.0, lang="fr"),
    Word(start_ts=2417408, end_ts=2420928, word=" was", probability=1.0, lang="fr"),
    Word(start_ts=2420928, end_ts=2422528, word=" considering", probability=1.0, lang="fr"),
    Word(start_ts=2422528, end_ts=2424448, word=" in", probability=1.0, lang="fr"),
    Word(start_ts=2424448, end_ts=2427968, word=" her", probability=1.0, lang="fr"),
    Word(start_ts=2429248, end_ts=2432768, word=" own", probability=1.0, lang="fr"),
    Word(start_ts=2432768, end_ts=2446208, word=" mind", probability=1.0, lang="fr"),
    Word(start_ts=2446208, end_ts=2448768, word=" (as", probability=1.0, lang="fr"),
    Word(start_ts=2448768, end_ts=2452288, word=" well", probability=1.0, lang="fr"),
    Word(start_ts=2452288, end_ts=2453888, word=" as", probability=1.0, lang="fr"),
    Word(start_ts=2453888, end_ts=2455168, word=" she", probability=1.0, lang="fr"),
    Word(start_ts=2455168, end_ts=2457728, word=" could,", probability=1.0, lang="fr"),
    Word(start_ts=2457728, end_ts=2458368, word=" for", probability=1.0, lang="fr"),
    Word(start_ts=2458368, end_ts=2459968, word=" the", probability=1.0, lang="fr"),
    Word(start_ts=2459968, end_ts=2461248, word=" hot", probability=1.0, lang="fr"),
    Word(start_ts=2461248, end_ts=2462848, word=" day", probability=1.0, lang="fr"),
    Word(start_ts=2462848, end_ts=2464768, word=" made", probability=1.0, lang="fr"),
    Word(start_ts=2464768, end_ts=2469248, word=" her", probability=1.0, lang="fr"),
    Word(start_ts=2469248, end_ts=2475968, word=" feel", probability=1.0, lang="fr"),
    Word(start_ts=2477888, end_ts=2496768, word=" very", probability=1.0, lang="fr"),
    Word(start_ts=2496768, end_ts=2499008, word=" sleepy", probability=1.0, lang="fr"),
    Word(start_ts=2499008, end_ts=2509888, word=" and", probability=1.0, lang="fr"),
    Word(start_ts=2509888, end_ts=2512448, word=" stupid),", probability=1.0, lang="fr"),
    Word(start_ts=2512448, end_ts=2515648, word=" whether", probability=1.0, lang="fr"),
    Word(start_ts=2515648, end_ts=2523008, word=" the", probability=1.0, lang="fr"),
]

WORDS_ULYSSES = [
    Word(start_ts=0, end_ts=0, word=f" {w}", probability=1.0, lang="en")
    for w in "Stately, plump Buck Mulligan came from the stairhead,"
    " bearing a bowl of lather on which a mirror and a razor lay crossed."
    " A yellow dressinggown, ungirdled, was sustained gently behind him on the mild morning air.".split()
]


class TestSentences(unittest.TestCase):
    # pylint: disable=invalid-name
    def test_SilenceSentenceTokenizer(self):
        # pylint: disable=import-outside-toplevel
        from verbatim.transcript.sentences import SilenceSentenceTokenizer

        sentence_tokenizer = SilenceSentenceTokenizer()

        sentences = sentence_tokenizer.split(words=WORDS_ALICE)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(
            sentences[0],
            " Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do:"
            " once or twice she had peeped into the book her sister was reading,",
        )
        self.assertEqual(
            sentences[1],
            " but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without"
            " pictures or conversations?” So she was considering in her own mind (as well as she could, for the hot"
            " day made her feel very sleepy and stupid), whether the",
        )

    def test_FastSentenceTokenizer(self):
        # pylint: disable=import-outside-toplevel
        from verbatim.transcript.sentences import FastSentenceTokenizer

        sentence_tokenizer = FastSentenceTokenizer()

        sentences = sentence_tokenizer.split(words=WORDS_ULYSSES)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(
            sentences[0], " Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed."
        )
        self.assertEqual(sentences[1], " A yellow dressinggown, ungirdled, was sustained gently behind him on the mild morning air.")

    @pytest.mark.slow
    @pytest.mark.requires_hf
    def test_SaTSentenceTokenizer(self):
        # pylint: disable=import-outside-toplevel
        from verbatim.transcript.sentences import SaTSentenceTokenizer

        sentence_tokenizer = SaTSentenceTokenizer(device="cpu")

        sentences = sentence_tokenizer.split(words=WORDS_ULYSSES)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(
            sentences[0],
            " Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed. ",
        )
        self.assertEqual(sentences[1], "A yellow dressinggown, ungirdled, was sustained gently behind him on the mild morning air.")


if __name__ == "__main__":
    unittest.main()

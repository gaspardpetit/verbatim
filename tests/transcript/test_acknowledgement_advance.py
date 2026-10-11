import unittest

import numpy as np

from verbatim.config import Config
from verbatim.transcript.words import Utterance, Word
from verbatim.verbatim import (
    SecondOpinionProbe,
    Verbatim,
    find_superseded_provisional_words,
    reconcile_second_opinion,
)
from verbatim_audio.sources.audiosource import AudioStream
from verbatim_transcript import TranscriberCapabilities


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


class ProbeTranscriber:
    def __init__(self, accepts_language_hint=True):
        self.accepts_language_hint = accepts_language_hint
        self.calls = []
        self.words = None

    def get_capabilities(self):
        return TranscriberCapabilities(
            accepts_language_hint=self.accepts_language_hint,
            detects_language=False,
            reports_language=False,
            supports_prompt=False,
            supports_prefix=False,
        )

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return self.words if self.words is not None else [make_word(0, 16000, " hello."), make_word(16500, 20000, " you")]


class StubVerbatim(Verbatim):
    def __init__(self, *, config, models, confirmed_words, unconfirmed_words):
        super().__init__(config=config, models=models)
        self._confirmed_words = confirmed_words
        self._unconfirmed_words = unconfirmed_words
        self.force_gap_mismatch = False

    def transcribe_window(self):
        return list(self._confirmed_words), list(self._unconfirmed_words)

    def gap_language_mismatch(self, *, start_ts, end_ts, expected_lang):
        _ = start_ts, end_ts, expected_lang
        return self.force_gap_mismatch


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
        verbatim.state.rolling_window.reset()
        return verbatim

    def test_post_acknowledgement_skip_uses_session_duration(self):
        hello = make_word(0, 16000, " hello.")
        for seconds, expected_ts in [(0.0, 16000), (0.1, 17600), (0.25, 20000)]:
            with self.subTest(seconds=seconds):
                verbatim = self._make_verbatim(
                    sentences=[" hello."], confirmed_words=[hello], unconfirmed_words=[],
                )
                verbatim.config.post_acknowledgement_skip_duration = seconds

                list(verbatim.process_audio_window(audio_stream=self.audio_stream))

                self.assertEqual(expected_ts, verbatim.state.acknowledged_ts)
                self.assertEqual(expected_ts, verbatim.state.window_ts)

    def test_language_mismatch_skip_uses_session_duration(self):
        hello = make_word(16000, 32000, " hello.")
        for seconds, expected_ts in [(0.1, 1600), (0.25, 4000)]:
            with self.subTest(seconds=seconds):
                verbatim = self._make_verbatim(sentences=[" hello."], confirmed_words=[hello], unconfirmed_words=[])
                verbatim.config.language_mismatch_skip_duration = seconds
                verbatim.force_gap_mismatch = True

                list(verbatim.process_audio_window(audio_stream=self.audio_stream))

                self.assertEqual(expected_ts, verbatim.state.window_ts)

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

    def test_second_opinion_observes_crossed_unconfirmed_word_without_changing_it(self):
        hello = make_word(0, 16000, " hello.")
        uncertain = make_word(16500, 20000, " you")
        verbatim = self._make_verbatim(
            sentences=[" hello."], confirmed_words=[hello], unconfirmed_words=[uncertain],
        )
        model = ProbeTranscriber()
        verbatim.models.transcriber = model
        verbatim.experimental_second_opinion = True

        list(verbatim.process_audio_window(audio_stream=self.audio_stream))

        self.assertEqual(1, len(verbatim.second_opinion_probes))
        self.assertEqual((0, 20000, "en"), (
            verbatim.second_opinion_probes[0].start_ts,
            verbatim.second_opinion_probes[0].end_ts,
            verbatim.second_opinion_probes[0].language,
        ))
        self.assertEqual(20000, len(model.calls[0]["audio"]))
        self.assertEqual("en", model.calls[0]["lang"])
        self.assertEqual("", model.calls[0]["prefix"])
        self.assertEqual([uncertain], verbatim.state.unconfirmed_words)
        self.assertEqual(16500, verbatim.state.acknowledged_ts)

    def test_second_opinion_omits_unsupported_language_hint(self):
        hello = make_word(0, 16000, " hello.")
        uncertain = make_word(16500, 20000, " you")
        verbatim = self._make_verbatim(
            sentences=[" hello."], confirmed_words=[hello], unconfirmed_words=[uncertain],
        )
        model = ProbeTranscriber(accepts_language_hint=False)
        verbatim.models.transcriber = model
        verbatim.experimental_second_opinion = True

        list(verbatim.process_audio_window(audio_stream=self.audio_stream))

        self.assertEqual(1, len(model.calls))
        self.assertIsNone(model.calls[0]["lang"])

    def test_second_opinion_reviews_old_provisional_word_between_acknowledged_utterances(self):
        first = Utterance.from_words("first", [make_word(0, 16000, " Nice to see you.")])
        second = Utterance.from_words("second", [make_word(21000, 36000, " You have a nice view.")])
        extra = make_word(17000, 20000, " You")
        verbatim = self._make_verbatim(sentences=[], confirmed_words=[], unconfirmed_words=[], audio_ts=50000)
        model = ProbeTranscriber()
        anchor_word = make_word(0, 16000, " Nice to see you.")
        boundary_word = make_word(15000, 16500, " edge")
        target_word = make_word(17000, 20000, " You")
        tail_word = make_word(23000, 25000, " uh")
        model.words = [anchor_word, boundary_word, target_word, tail_word]
        verbatim.models.transcriber = model
        verbatim.state.acknowledged_utterances = [first, second]
        verbatim._observed_unconfirmed_words = [extra]  # pylint: disable=protected-access

        verbatim.probe_second_opinion(second, proposed_ack_ts=37600)

        self.assertEqual(1, len(verbatim.second_opinion_probes))
        probe = verbatim.second_opinion_probes[0]
        self.assertEqual((0, 36000, first, second, [extra]),
                         (probe.start_ts, probe.end_ts, probe.acknowledged, probe.following, probe.uncertain_words))
        self.assertEqual("en", model.calls[0]["lang"])
        self.assertEqual(36000, len(model.calls[0]["audio"]))
        self.assertEqual((16000, 21000), (probe.target_start_ts, probe.target_end_ts))
        self.assertEqual([target_word], probe.target_words)
        self.assertEqual([boundary_word], probe.boundary_words)
        self.assertNotIn(tail_word, probe.target_words + probe.boundary_words)

    def test_second_opinion_reviews_current_provisional_word_before_new_acknowledgement(self):
        first = Utterance.from_words("first", [make_word(0, 16000, " Done.")])
        second = Utterance.from_words("second", [make_word(21000, 28000, " Next")])
        provisional = make_word(19000, 20500, " Next")
        verbatim = self._make_verbatim(sentences=[], confirmed_words=[], unconfirmed_words=[], audio_ts=30000)
        model = ProbeTranscriber()
        model.words = [make_word(0, 16000, " Done."), make_word(19000, 20500, " Next")]
        verbatim.models.transcriber = model
        verbatim.state.acknowledged_utterances = [first, second]
        verbatim.state.unconfirmed_words = [provisional]

        verbatim.probe_second_opinion(second, proposed_ack_ts=29600)

        self.assertEqual(1, len(verbatim.second_opinion_probes))
        self.assertEqual([provisional], verbatim.second_opinion_probes[0].uncertain_words)
        resolution = verbatim.second_opinion_probes[0].resolution
        if resolution is None:
            self.fail("Expected a second-opinion resolution")
        self.assertEqual([provisional], resolution.redundant_provisional_words)

    def test_second_opinion_skips_word_already_superseded_by_confirmation(self):
        first = Utterance.from_words("first", [make_word(0, 16000, " Done.")])
        second = Utterance.from_words("second", [make_word(21000, 28000, " Next")])
        provisional = make_word(19000, 20500, " Next")
        verbatim = self._make_verbatim(sentences=[], confirmed_words=[], unconfirmed_words=[], audio_ts=30000)
        model = ProbeTranscriber()
        verbatim.models.transcriber = model
        verbatim.state.acknowledged_utterances = [first, second]
        verbatim.state.unconfirmed_words = [provisional]
        verbatim.superseded_provisional_words = [provisional]

        verbatim.probe_second_opinion(second, proposed_ack_ts=29600)

        self.assertEqual([], verbatim.second_opinion_probes)
        self.assertEqual([], model.calls)

    def test_second_opinion_aligns_both_anchors_before_marking_provisional_redundant(self):
        previous = Utterance.from_words("previous", [make_word(10000, 11000, " see"), make_word(11000, 12000, " you.")])
        following = Utterance.from_words("following", [make_word(14000, 15000, " You"), make_word(15000, 16000, " have")])
        provisional = make_word(13500, 14500, " You")
        model_words = [
            make_word(10100, 11100, " see"), make_word(11100, 12100, " you."),
            make_word(13600, 14600, " You"), make_word(14600, 15600, " have"),
        ]
        probe = SecondOpinionProbe(10000, 16000, 12000, 14000, "en", previous, following, [provisional], model_words, [], [])

        resolution = reconcile_second_opinion(probe)

        self.assertEqual("redundant_provisional", resolution.status)
        self.assertEqual([provisional], resolution.redundant_provisional_words)
        self.assertEqual([], resolution.model_gap_words)

        extra_spoken_word = make_word(12800, 13400, " You")
        probe.model_words = model_words[:2] + [extra_spoken_word] + model_words[2:]
        resolution = reconcile_second_opinion(probe)
        self.assertEqual("additional_speech", resolution.status)
        self.assertEqual([extra_spoken_word], resolution.model_gap_words)
        self.assertEqual([], resolution.redundant_provisional_words)

        probe.model_words = model_words
        probe.uncertain_words = [make_word(12500, 13500, " You")]
        resolution = reconcile_second_opinion(probe)
        self.assertEqual("inconclusive", resolution.status)
        self.assertEqual([], resolution.redundant_provisional_words)

    def test_second_opinion_rejects_changed_acknowledged_anchor(self):
        previous = Utterance.from_words("previous", [make_word(10000, 11000, " Hey.")])
        following = Utterance.from_words("following", [make_word(14000, 15000, " Good.")])
        provisional = make_word(12000, 13000, " Good.")
        model_words = [make_word(10000, 11000, " A"), make_word(14000, 15000, " good.")]
        probe = SecondOpinionProbe(10000, 15000, 11000, 14000, "en", previous, following, [provisional], model_words, [], [])

        resolution = reconcile_second_opinion(probe)

        self.assertEqual("inconclusive", resolution.status)
        self.assertEqual([], resolution.redundant_provisional_words)

    def test_second_opinion_keeps_real_gap_word_while_resolving_duplicate_prefix(self):
        previous = Utterance.from_words("previous", [make_word(10000, 11000, " done.")])
        following = Utterance.from_words("following", [make_word(15000, 16000, " You"), make_word(16000, 17000, " know")])
        real_gap = make_word(12500, 13500, " And")
        duplicate = make_word(14800, 15800, " You")
        model_words = [
            make_word(10000, 11000, " done."), make_word(12500, 13500, " And"),
            make_word(14800, 15800, " You"), make_word(15800, 16800, " know"),
        ]
        probe = SecondOpinionProbe(10000, 17000, 11000, 15000, "en", previous, following,
                                   [real_gap, duplicate], model_words, [], [])

        resolution = reconcile_second_opinion(probe)

        self.assertEqual("redundant_provisional", resolution.status)
        self.assertEqual([real_gap], resolution.model_gap_words)
        self.assertEqual([duplicate], resolution.redundant_provisional_words)

    def test_second_opinion_does_not_drop_filler_omitted_by_model(self):
        previous = Utterance.from_words("previous", [make_word(10000, 11000, " done.")])
        following = Utterance.from_words("following", [make_word(15000, 16000, " Next")])
        filler = make_word(13000, 14000, " um")
        probe = SecondOpinionProbe(10000, 16000, 11000, 15000, "en", previous, following,
                                   [filler], [make_word(10000, 11000, " done."), make_word(15000, 16000, " Next")], [], [])

        resolution = reconcile_second_opinion(probe)

        self.assertEqual("inconclusive", resolution.status)
        self.assertEqual([], resolution.redundant_provisional_words)

    def test_confirmation_lineage_retires_shifted_provisional_prefix(self):
        older = [
            make_word(10900, 11060, " You"), make_word(11060, 11220, " have"),
            make_word(11220, 11300, " a"), make_word(11300, 11620, " nice"),
        ]
        confirmed = [
            make_word(11080, 11160, "You"), make_word(11160, 11320, " have"),
            make_word(11320, 11400, " a"), make_word(11400, 11640, " nice"),
        ]

        superseded = find_superseded_provisional_words([older[0]], [older], confirmed)

        self.assertEqual([older[0]], superseded)

    def test_confirmation_lineage_preserves_genuine_repeat_with_missing_model_word(self):
        first = make_word(60400, 62800, " Good,")
        second = make_word(65200, 67600, " good.")
        older = [first, second, make_word(69800, 71400, " how"), make_word(71400, 73800, " are")]
        confirmed = [
            make_word(65200, 67600, "Good."), make_word(69800, 71400, " how"),
            make_word(71400, 73800, " are"),
        ]

        superseded = find_superseded_provisional_words([first], [older], confirmed)

        self.assertEqual([], superseded)


if __name__ == "__main__":
    unittest.main()

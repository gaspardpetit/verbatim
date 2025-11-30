import unittest

import numpy as np

from verbatim_core import LanguageDetectionRequest, TranscriptionWindowResult, detect_language


class TestCoreLanguage(unittest.TestCase):
    def test_detect_language_defaults_to_en_when_lang_empty(self):
        req = LanguageDetectionRequest(audio=np.zeros(16000, dtype=np.float32), lang=[], timestamp=0, window_ts=0, audio_ts=16000)
        result = detect_language(request=req, guess_fn=lambda _audio, _langs: ("xx", 0.0))
        self.assertEqual(result.language, "en")
        self.assertEqual(result.probability, 1.0)
        self.assertEqual(result.samples_used, 0)

    def test_detect_language_respects_single_language_hint(self):
        req = LanguageDetectionRequest(audio=np.zeros(16000, dtype=np.float32), lang=["fr"], timestamp=0, window_ts=0, audio_ts=16000)
        result = detect_language(request=req, guess_fn=lambda _audio, _langs: ("en", 0.9))
        self.assertEqual(result.language, "fr")
        self.assertEqual(result.probability, 1.0)
        self.assertEqual(result.samples_used, 0)

    def test_detect_language_expands_window_until_confident(self):
        # Prepare audio longer than initial 2s window
        total_samples = 16000 * 8
        audio = np.arange(total_samples, dtype=np.float32)
        calls = []

        def guess_fn(chunk, langs):
            calls.append((len(chunk), tuple(langs)))
            # First call low confidence, second call confident
            if len(calls) == 1:
                return ("en", 0.2)
            return ("es", 0.8)

        req = LanguageDetectionRequest(audio=audio, lang=["en", "es"], timestamp=0, window_ts=0, audio_ts=total_samples)
        result = detect_language(request=req, guess_fn=guess_fn)

        # First try uses 2s (32k samples), second uses 4s (64k)
        self.assertEqual(calls[0][0], 32000)
        self.assertEqual(calls[1][0], 64000)
        self.assertEqual(result.language, "es")
        self.assertEqual(result.samples_used, 64000)

    def test_transcription_window_result_as_tuple_lists_sequences(self):
        result = TranscriptionWindowResult(utterance="u", unacknowledged=("a1", "a2"), unconfirmed_words=("w1",))
        utterance, unack, unconfirmed = result.as_tuple()
        self.assertEqual(utterance, "u")
        self.assertIsInstance(unack, list)
        self.assertEqual(unack, ["a1", "a2"])
        self.assertIsInstance(unconfirmed, list)
        self.assertEqual(unconfirmed, ["w1"])


if __name__ == "__main__":
    unittest.main()

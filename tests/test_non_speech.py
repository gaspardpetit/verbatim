# pylint: disable=protected-access
import unittest
from unittest.mock import patch

import numpy as np

from verbatim.config import Config
from verbatim.non_speech import AstNonSpeechClassifier
from verbatim.verbatim import Verbatim


class DummyModels:
    pass


class _FakeClassifier:
    def classify(self, segment, sample_rate):
        _ = segment, sample_rate
        return ["music"]


class TestNonSpeechClassifierIntegration(unittest.TestCase):
    def test_skip_marker_uses_ast_backend_when_enabled(self):
        config = Config(device="cpu")
        config.non_speech_backend = "ast"
        verbatim = Verbatim(
            config=config,
            models=DummyModels(),
            vad_callback=lambda audio, min_speech_duration_ms, min_silence_duration_ms: [],
        )
        verbatim.state.window_ts = 0
        verbatim.state.audio_ts = 100000
        verbatim.state.rolling_window.array = np.full(len(verbatim.state.rolling_window.array), 0.05, dtype=np.float32)

        with patch("verbatim.verbatim.create_non_speech_classifier", return_value=_FakeClassifier()):
            result = verbatim.skip_leading_silence(max_skip=100000)

        self.assertIsNotNone(result.marker)
        marker = result.marker
        if marker is None:
            self.fail("Expected a marker utterance")
        self.assertEqual("[MUSIC]", marker.text)


class TestAstNonSpeechClassifier(unittest.TestCase):
    def test_scores_to_labels_maps_known_classes(self):
        classifier = AstNonSpeechClassifier.__new__(AstNonSpeechClassifier)
        classifier._model = type(
            "FakeModel",
            (),
            {"config": type("FakeConfig", (), {"id2label": {0: "Music", 1: "Typing", 2: "Unknown"}})()},
        )()

        labels = classifier._scores_to_labels(np.array([0.7, 0.4, 0.8], dtype=np.float32))

        self.assertEqual(["music", "mechanical_noise"], labels)

    def test_chunk_audio_includes_trailing_tail(self):
        audio = np.zeros(int(6.25 * 16000), dtype=np.float32)

        chunks = AstNonSpeechClassifier._chunk_audio(audio)

        self.assertEqual(2, len(chunks))
        self.assertEqual(int(5.0 * 16000), len(chunks[-1]))


if __name__ == "__main__":
    unittest.main()

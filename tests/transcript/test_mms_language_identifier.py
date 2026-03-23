import sys
import types
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

import verbatim.language_id as language_id_module
from verbatim.config import Config
from verbatim.language_id import MmsLanguageIdentifier, TranscriberLanguageIdentifier, create_language_identifier


class DummyTranscriber:
    def guess_language(self, audio, lang):
        _ = audio
        return lang[-1], 0.75


class DummyModels:
    def __init__(self):
        self.transcriber = DummyTranscriber()


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def to(self, _device):
        return self

    def __getitem__(self, index):
        return FakeTensor(self._values[index])

    def tolist(self):
        return self._values


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type, exc, tb
        return False


class FakeFeatureExtractor:
    @classmethod
    def from_pretrained(cls, _model_size_or_path):
        return cls()

    def __call__(self, audio, sampling_rate, return_tensors):
        _ = audio, sampling_rate, return_tensors
        return {"input_values": FakeTensor([[0.0]])}


class FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(id2label={0: "eng", 1: "fra", 2: "deu"})

    @classmethod
    def from_pretrained(cls, _model_size_or_path):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return None

    def __call__(self, **_inputs):
        return FakeModelOutput(logits=FakeTensor([[0.1, 0.8, 0.2]]))


class TestMmsLanguageIdentifier(unittest.TestCase):
    def test_transcriber_identifier_delegates(self):
        identifier = TranscriberLanguageIdentifier(models=DummyModels())
        language, probability = identifier.guess_language(audio=np.zeros(10, dtype=np.float32), lang=["en", "fr"])
        self.assertEqual("fr", language)
        self.assertEqual(0.75, probability)

    def test_mms_identifier_picks_best_allowed_language(self):
        fake_transformers: Any = types.ModuleType("transformers")
        fake_transformers_auto: Any = types.ModuleType("transformers.models.auto.feature_extraction_auto")
        fake_transformers_auto.AutoFeatureExtractor = FakeFeatureExtractor
        fake_transformers_wav2vec2: Any = types.ModuleType("transformers.models.wav2vec2")
        fake_transformers_wav2vec2.Wav2Vec2ForSequenceClassification = FakeModel

        fake_torch: Any = types.ModuleType("torch")
        fake_torch.no_grad = FakeNoGrad
        fake_torch.nn = types.SimpleNamespace(
            functional=types.SimpleNamespace(
                softmax=lambda logits, dim=-1: logits,
            )
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "transformers": fake_transformers,
                    "transformers.models.auto.feature_extraction_auto": fake_transformers_auto,
                    "transformers.models.wav2vec2": fake_transformers_wav2vec2,
                },
            ),
            patch.object(language_id_module, "torch", fake_torch),
        ):
            identifier = MmsLanguageIdentifier(model_size_or_path="facebook/mms-lid-126", device="cpu")
            language, probability = identifier.guess_language(audio=np.zeros(16000, dtype=np.float32), lang=["en", "fr"])

        self.assertEqual("fr", language)
        self.assertEqual(0.8, probability)

    def test_create_language_identifier_from_config(self):
        config = Config(device="cpu")
        config.language_identifier_backend = "transcriber"
        identifier = create_language_identifier(config=config, models=DummyModels())
        self.assertIsInstance(identifier, TranscriberLanguageIdentifier)


if __name__ == "__main__":
    unittest.main()

# pylint: disable=protected-access
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from verbatim.voices.transcribe.qwen_asr import QwenAsrTranscriber


class FakeTimestamp:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class FakeResult:
    def __init__(self, language, text, time_stamps=None):
        self.language = language
        self.text = text
        self.time_stamps = time_stamps


class FakeLoadedModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


class FakeQwen3ASRModel:
    loaded_model = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        _ = args, kwargs
        return cls.loaded_model


class TestQwenAsrTranscriber(unittest.TestCase):
    def _make_transcriber(self, results):
        fake_model = FakeLoadedModel(results=results)
        fake_qwen_module = types.ModuleType("qwen_asr")
        fake_qwen_module.Qwen3ASRModel = FakeQwen3ASRModel
        FakeQwen3ASRModel.loaded_model = fake_model

        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float32 = "float32"

        module_overrides = {
            "qwen_asr": fake_qwen_module,
            "torch": fake_torch,
        }
        with patch.dict(sys.modules, module_overrides):
            transcriber = QwenAsrTranscriber(
                model_size_or_path="Qwen/Qwen3-ASR-1.7B",
                aligner_model_size_or_path="Qwen/Qwen3-ForcedAligner-0.6B",
                device="cuda",
            )

        return transcriber, fake_model

    def test_guess_language_maps_qwen_name_to_iso_code(self):
        transcriber, fake_model = self._make_transcriber(
            results=[
                [FakeResult(language="English", text="hello")],
            ]
        )

        language, probability = transcriber.guess_language(np.zeros(16000, dtype=np.float32), ["en", "fr"])

        self.assertEqual("en", language)
        self.assertEqual(1.0, probability)
        self.assertIsNone(fake_model.calls[0]["language"])
        self.assertFalse(fake_model.calls[0]["return_time_stamps"])

    def test_transcribe_uses_prefix_as_context_and_returns_words(self):
        transcriber, fake_model = self._make_transcriber(
            results=[
                [
                    FakeResult(
                        language="English",
                        text=" hello world",
                        time_stamps=[
                            FakeTimestamp("hello", 0.0, 0.4),
                            FakeTimestamp("world", 0.4, 0.8),
                        ],
                    )
                ],
            ]
        )

        words = transcriber.transcribe(
            audio=np.zeros(16000, dtype=np.float32),
            lang="en",
            prompt="This is a sentence.",
            prefix=" hello",
            window_ts=3200,
            audio_ts=16000,
        )

        self.assertEqual(" hello world", "".join(word.word for word in words))
        self.assertEqual(" hello", fake_model.calls[0]["context"])
        self.assertEqual("English", fake_model.calls[0]["language"])
        self.assertEqual(3200, words[0].start_ts)
        self.assertEqual("en", words[0].lang)

    def test_transcribe_requires_timestamps(self):
        transcriber, _fake_model = self._make_transcriber(
            results=[
                [FakeResult(language="English", text=" hello world", time_stamps=None)],
            ]
        )

        with self.assertRaises(RuntimeError):
            transcriber.transcribe(
                audio=np.zeros(16000, dtype=np.float32),
                lang="en",
                prompt="This is a sentence.",
                prefix="",
                window_ts=0,
                audio_ts=16000,
            )

    def test_split_transcript_text_preserves_exact_spacing(self):
        tokens = QwenAsrTranscriber._split_transcript_text("Toto, I have\n a feeling")
        self.assertEqual(["Toto,", " I", " have", "\n a", " feeling"], tokens)

    def test_project_timestamps_onto_transcript_preserves_original_text(self):
        words = QwenAsrTranscriber._project_timestamps_onto_transcript(
            transcript_text="Toto, I have a feeling we're not in Kansas anymore.",
            aligned_units=[
                (0, 10, "Toto", 1.0),
                (10, 20, ", I", 1.0),
                (20, 30, "have", 1.0),
                (30, 40, "a", 1.0),
                (40, 50, "feeling", 1.0),
                (50, 60, "we're", 1.0),
                (60, 70, "not", 1.0),
                (70, 80, "in", 1.0),
                (80, 90, "Kansas", 1.0),
                (90, 100, "anymore", 1.0),
            ],
            lang="en",
            window_ts=0,
            audio_ts=200,
        )
        self.assertEqual("Toto, I have a feeling we're not in Kansas anymore.", "".join(word.word for word in words))
        self.assertEqual("Toto,", words[0].word)
        self.assertEqual(" I", words[1].word)
        self.assertEqual(" anymore.", words[-1].word)

    def test_project_timestamps_skips_units_beyond_audio_end(self):
        words = QwenAsrTranscriber._project_timestamps_onto_transcript(
            transcript_text=" hello world",
            aligned_units=[
                (0, 100, "hello", 1.0),
                (100, 200, "world", 1.0),
            ],
            lang="en",
            window_ts=0,
            audio_ts=150,
        )
        self.assertEqual(" hello", "".join(word.word for word in words))
        self.assertEqual(1, len(words))


if __name__ == "__main__":
    unittest.main()

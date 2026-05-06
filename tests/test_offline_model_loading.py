# pylint: disable=unsubscriptable-object,unpacking-non-sequence
import os
import sys
import types
import unittest
from typing import Any, cast
from unittest.mock import patch

import verbatim.language_id as language_id_module
import verbatim.model_cache as model_cache_module
import verbatim.non_speech as non_speech_module
import verbatim.transcript.sentences as sentences_module
from verbatim.language_id import MmsLanguageIdentifier
from verbatim.non_speech import AstNonSpeechClassifier
from verbatim.transcript.sentences import SaTSentenceTokenizer


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def to(self, _device):
        return self

    def __getitem__(self, index):
        return _FakeTensor(self._values[index])

    def tolist(self):
        return self._values


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type, exc, tb
        return False


class _FakeFeatureExtractor:
    last_call = None

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        cls.last_call = (model, kwargs)
        return cls()

    def __call__(self, *_args, **_kwargs):
        return {"input_values": _FakeTensor([[0.0]])}


class _FakeLanguageModel:
    last_call = None

    def __init__(self):
        self.config = types.SimpleNamespace(id2label={0: "eng"})

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        cls.last_call = (model, kwargs)
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return None


class _FakeAudioModel:
    last_call = None

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        cls.last_call = (model, kwargs)
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return None

    @property
    def config(self):
        return types.SimpleNamespace(problem_type=None)


class _FakeSaT:
    last_call = None

    def __init__(self, *args, **kwargs):
        _FakeSaT.last_call = (args, kwargs)

    def half(self):
        return self

    def to(self, _device):
        return self


class TestOfflineModelLoading(unittest.TestCase):
    def test_resolve_hf_snapshot_path_raises_clear_error_offline(self):
        with patch.dict(os.environ, {"VERBATIM_OFFLINE": "1"}, clear=False):
            with patch.object(model_cache_module, "_snapshot_download", side_effect=RuntimeError("missing")):
                with self.assertRaisesRegex(RuntimeError, "Offline mode is enabled and test model 'acme/model' is not installed"):
                    model_cache_module.resolve_hf_snapshot_path("acme/model", purpose="test model", cache_dir="D:/models")

    def test_resolve_hf_file_path_raises_clear_error_offline(self):
        with patch.dict(os.environ, {"VERBATIM_OFFLINE": "1"}, clear=False):
            with patch.object(model_cache_module, "_hf_hub_download", side_effect=RuntimeError("missing")):
                with self.assertRaisesRegex(RuntimeError, "Offline mode is enabled and pyannote model 'acme/model' is not installed"):
                    model_cache_module.resolve_hf_file_path(
                        "acme/model",
                        filename="config.yaml",
                        purpose="pyannote model",
                        cache_dir="D:/models/pyannote",
                    )

    def test_mms_uses_local_files_only_when_offline(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = _FakeNoGrad
        fake_torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(softmax=lambda logits, dim=-1: logits))
        fake_transformers_auto = types.ModuleType("transformers.models.auto.feature_extraction_auto")
        fake_transformers_auto.AutoFeatureExtractor = _FakeFeatureExtractor
        fake_transformers_wav2vec2 = types.ModuleType("transformers.models.wav2vec2")
        fake_transformers_wav2vec2.Wav2Vec2ForSequenceClassification = _FakeLanguageModel

        with (
            patch.dict(os.environ, {"VERBATIM_OFFLINE": "1", "HUGGINGFACE_HUB_CACHE": "D:/cache/hf"}, clear=False),
            patch.dict(
                sys.modules,
                {
                    "transformers.models.auto.feature_extraction_auto": fake_transformers_auto,
                    "transformers.models.wav2vec2": fake_transformers_wav2vec2,
                },
            ),
            patch.object(language_id_module, "torch", fake_torch),
            patch.object(language_id_module, "resolve_hf_snapshot_path", return_value="D:/cache/mms"),
        ):
            MmsLanguageIdentifier(model_size_or_path="facebook/mms-lid-126", device="cpu")

        self.assertIsNotNone(_FakeFeatureExtractor.last_call)
        self.assertIsNotNone(_FakeLanguageModel.last_call)
        feature_call = cast(tuple[Any, dict[str, Any]], _FakeFeatureExtractor.last_call)
        model_call = cast(tuple[Any, dict[str, Any]], _FakeLanguageModel.last_call)
        self.assertEqual(feature_call[0], "D:/cache/mms")
        self.assertTrue(feature_call[1]["local_files_only"])
        self.assertEqual(feature_call[1]["cache_dir"], "D:/cache/hf")
        self.assertTrue(model_call[1]["local_files_only"])

    def test_ast_uses_local_files_only_when_offline(self):
        fake_torch = types.ModuleType("torch")
        fake_torchaudio = types.ModuleType("torchaudio")
        fake_transformers_auto_feature = types.ModuleType("transformers.models.auto.feature_extraction_auto")
        fake_transformers_auto_feature.AutoFeatureExtractor = _FakeFeatureExtractor
        fake_transformers_auto_model = types.ModuleType("transformers.models.auto.modeling_auto")
        fake_transformers_auto_model.AutoModelForAudioClassification = _FakeAudioModel

        with (
            patch.dict(os.environ, {"VERBATIM_OFFLINE": "1", "HUGGINGFACE_HUB_CACHE": "D:/cache/hf"}, clear=False),
            patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "torchaudio": fake_torchaudio,
                    "transformers.models.auto.feature_extraction_auto": fake_transformers_auto_feature,
                    "transformers.models.auto.modeling_auto": fake_transformers_auto_model,
                },
            ),
            patch.object(non_speech_module, "resolve_hf_snapshot_path", return_value="D:/cache/ast"),
        ):
            AstNonSpeechClassifier(model_name="MIT/ast-finetuned-audioset-10-10-0.4593", device="cpu")

        self.assertIsNotNone(_FakeFeatureExtractor.last_call)
        self.assertIsNotNone(_FakeAudioModel.last_call)
        feature_call = cast(tuple[Any, dict[str, Any]], _FakeFeatureExtractor.last_call)
        model_call = cast(tuple[Any, dict[str, Any]], _FakeAudioModel.last_call)
        self.assertEqual(feature_call[0], "D:/cache/ast")
        self.assertTrue(feature_call[1]["local_files_only"])
        self.assertTrue(model_call[1]["local_files_only"])

    def test_sat_uses_local_paths_and_offline_kwargs(self):
        fake_wtpsplit = types.ModuleType("wtpsplit")
        fake_wtpsplit.SaT = _FakeSaT

        with (
            patch.dict(os.environ, {"VERBATIM_OFFLINE": "1", "HUGGINGFACE_HUB_CACHE": "D:/cache/hf"}, clear=False),
            patch.dict(sys.modules, {"wtpsplit": fake_wtpsplit}),
            patch.object(
                sentences_module,
                "resolve_hf_snapshot_path",
                side_effect=["D:/cache/sat-model", "D:/cache/tokenizer"],
            ),
        ):
            SaTSentenceTokenizer(device="cpu")

        self.assertIsNotNone(_FakeSaT.last_call)
        args, kwargs = cast(tuple[tuple[Any, ...], dict[str, Any]], _FakeSaT.last_call)
        self.assertEqual(args[0], "D:/cache/sat-model")
        self.assertEqual(kwargs["tokenizer_name_or_path"], "D:/cache/tokenizer")
        self.assertTrue(kwargs["from_pretrained_kwargs"]["local_files_only"])

    def test_sat_does_not_double_prefix_segment_any_text_repo(self):
        fake_wtpsplit = types.ModuleType("wtpsplit")
        fake_wtpsplit.SaT = _FakeSaT

        with (
            patch.dict(sys.modules, {"wtpsplit": fake_wtpsplit}),
            patch.object(
                sentences_module,
                "resolve_hf_snapshot_path",
                side_effect=["D:/cache/sat-model", "D:/cache/tokenizer"],
            ) as resolve_snapshot,
        ):
            SaTSentenceTokenizer(device="cpu", model="segment-any-text/sat-3l-sm")

        self.assertEqual(resolve_snapshot.call_args_list[0].args[0], "segment-any-text/sat-3l-sm")

    def test_sat_collapses_already_doubled_segment_any_text_repo(self):
        fake_wtpsplit = types.ModuleType("wtpsplit")
        fake_wtpsplit.SaT = _FakeSaT

        with (
            patch.dict(sys.modules, {"wtpsplit": fake_wtpsplit}),
            patch.object(
                sentences_module,
                "resolve_hf_snapshot_path",
                side_effect=["D:/cache/sat-model", "D:/cache/tokenizer"],
            ) as resolve_snapshot,
        ):
            SaTSentenceTokenizer(device="cpu", model="segment-any-text/segment-any-text/sat-3l-sm")

        self.assertEqual(resolve_snapshot.call_args_list[0].args[0], "segment-any-text/sat-3l-sm")


if __name__ == "__main__":
    unittest.main()

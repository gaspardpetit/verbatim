# pylint: disable=protected-access
import builtins
import sys
import types
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

from verbatim.models import Models


class _FakeFasterWhisperTranscriber:
    def __init__(self, *, model_size_or_path, device):
        self.model_size_or_path = model_size_or_path
        self.device = device


class TestModels(unittest.TestCase):
    def test_parakeet_openvino_backend_uses_selected_model(self):
        models = Models.__new__(Models)
        models._config_transcriber_backend = "parakeet-openvino"
        models._parakeet_model_size = "istupakov/parakeet-tdt-0.6b-v3-onnx"
        models._device = "cpu"

        with patch("verbatim.voices.transcribe.parakeet_openvino.ParakeetOpenVinoTranscriber") as transcriber_class:
            transcriber = models._build_transcriber()

        transcriber_class.assert_called_once_with(model_size_or_path=models._parakeet_model_size, device="cpu")
        self.assertIs(transcriber, transcriber_class.return_value)

    def test_parakeet_backend_uses_selected_checkpoint(self):
        models = Models.__new__(Models)
        models._config_transcriber_backend = "parakeet"
        models._parakeet_model_size = "nvidia/parakeet-tdt-0.6b-v2"
        models._device = "cuda"

        with patch("verbatim.voices.transcribe.parakeet.ParakeetTranscriber") as transcriber_class:
            transcriber = models._build_transcriber()

        transcriber_class.assert_called_once_with(model_size_or_path="nvidia/parakeet-tdt-0.6b-v2", device="cuda")
        self.assertIs(transcriber, transcriber_class.return_value)

    def test_missing_default_backend_has_install_hint(self):
        models = Models.__new__(Models)
        models._config_transcriber_backend = "auto"
        models._whisper_model_size = "large-v3"
        models._device = "cpu"

        missing_dependency = ModuleNotFoundError("No module named 'faster_whisper'", name="faster_whisper")
        with patch("sys.platform", "linux"):
            with patch.dict(Models._build_transcriber.__globals__, {"import_module": MagicMock(side_effect=missing_dependency)}):
                with self.assertRaisesRegex(RuntimeError, "verbatim\\[faster_whisper\\]"):
                    models._build_transcriber()

    def test_build_transcriber_falls_back_when_mlx_is_unavailable(self):
        models = Models.__new__(Models)
        models._config_transcriber_backend = "auto"
        models._whisper_model_size = "large-v3"
        models._device = "mps"
        models._qwen_asr_model_size = ""
        models._qwen_aligner_model_size = ""
        models._qwen_dtype = "auto"
        models._qwen_max_inference_batch_size = 1
        models._qwen_max_new_tokens = 256

        fake_fw_module = cast(Any, types.ModuleType("verbatim.voices.transcribe.faster_whisper"))
        fake_fw_module.FasterWhisperTranscriber = _FakeFasterWhisperTranscriber

        original_import = builtins.__import__

        def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            if name.endswith("whispermlx"):
                raise ImportError("mlx-whisper not installed")
            return original_import(name, globals_, locals_, fromlist, level)

        with patch("sys.platform", "darwin"):
            with patch("platform.processor", return_value="arm"):
                with patch.dict(sys.modules, {"verbatim.voices.transcribe.faster_whisper": fake_fw_module}):
                    with patch("builtins.__import__", side_effect=fake_import):
                        transcriber = models._build_transcriber()

        self.assertEqual(_FakeFasterWhisperTranscriber, transcriber.__class__)


if __name__ == "__main__":
    unittest.main()

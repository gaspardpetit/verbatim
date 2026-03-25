# pylint: disable=protected-access
import builtins
import sys
import types
import unittest
from unittest.mock import patch

from verbatim.models import Models


class _FakeFasterWhisperTranscriber:
    def __init__(self, *, model_size_or_path, device):
        self.model_size_or_path = model_size_or_path
        self.device = device


class TestModels(unittest.TestCase):
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

        fake_fw_module = types.ModuleType("verbatim.voices.transcribe.faster_whisper")
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

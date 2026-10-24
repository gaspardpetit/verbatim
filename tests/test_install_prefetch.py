import os
import tempfile
import unittest
from unittest.mock import patch

from verbatim.config import Config
from verbatim.prefetch import apply_cache_env, collect_install_requirements
from verbatim_audio.sources.sourceconfig import SourceConfig


class TestInstallRequirements(unittest.TestCase):
    def test_default_whisper_install_includes_sat_and_whisper(self):
        config = Config(device="cpu")
        source_config = SourceConfig()

        requirements = collect_install_requirements(config=config, source_config=source_config)

        self.assertTrue(requirements.include_sat)
        self.assertTrue(requirements.include_faster_whisper)
        self.assertFalse(requirements.include_qwen_asr)
        self.assertFalse(requirements.include_voxtral)
        self.assertFalse(requirements.include_mms_language_model)

    def test_qwen_mms_ast_install_includes_transitive_models(self):
        config = Config(device="cpu")
        config.transcriber_backend = "qwen"
        config.language_identifier_backend = "mms"
        config.non_speech_backend = "ast"
        source_config = SourceConfig()

        requirements = collect_install_requirements(config=config, source_config=source_config)

        self.assertTrue(requirements.include_sat)
        self.assertTrue(requirements.include_qwen_asr)
        self.assertTrue(requirements.include_mms_language_model)
        self.assertTrue(requirements.include_ast_noise_model)
        self.assertFalse(requirements.include_faster_whisper)

    def test_voxtral_transcriber_backend_pulls_in_mms_fallback(self):
        config = Config(device="cpu")
        config.transcriber_backend = "voxtral"
        config.language_identifier_backend = "transcriber"
        source_config = SourceConfig()

        requirements = collect_install_requirements(config=config, source_config=source_config)

        self.assertTrue(requirements.include_voxtral)
        self.assertTrue(requirements.include_mms_language_model)
        self.assertTrue(requirements.include_sat)

    def test_parakeet_install_selects_nemo_checkpoint(self):
        config = Config(device="cpu")
        config.transcriber_backend = "parakeet"

        requirements = collect_install_requirements(config=config, source_config=SourceConfig())

        self.assertTrue(requirements.include_parakeet)
        self.assertFalse(requirements.include_faster_whisper)

    def test_parakeet_openvino_install_selects_prepared_npu_blobs(self):
        with patch.object(Config, "is_device_supported", return_value=True):
            config = Config(device="cpu")
        config.transcriber_backend = "parakeet-openvino"

        requirements = collect_install_requirements(config=config, source_config=SourceConfig())

        self.assertTrue(requirements.include_parakeet_openvino)
        self.assertFalse(requirements.include_parakeet)
        self.assertFalse(requirements.include_faster_whisper)

    def test_offline_config_accepts_prepared_read_only_model_cache(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with patch.object(Config, "is_device_supported", return_value=True):
                with patch("verbatim.config.os.makedirs", side_effect=AssertionError("unexpected model-cache write")):
                    with patch("verbatim.config.os.access", return_value=False):
                        with patch.dict(os.environ, {}, clear=True):
                            config = Config(
                                device="cpu", offline=True, model_cache_dir=model_dir,
                                working_dir=None, output_dir=model_dir,
                            )
                            self.assertEqual(model_dir, os.environ["VERBATIM_MODELDIR"])
                            self.assertEqual("1", os.environ["HF_HUB_OFFLINE"])
                            self.assertIsNone(config.working_dir)

    def test_diarize_policy_adds_pyannote_and_separation(self):
        config = Config(device="cpu")
        source_config = SourceConfig(diarize_strategy="1,2=energy;3=pyannote;*=separate")

        requirements = collect_install_requirements(config=config, source_config=source_config)

        self.assertTrue(requirements.include_pyannote_diarization)
        self.assertTrue(requirements.include_pyannote_separation)

    def test_stream_mode_skips_sentence_tokenizer_prefetch(self):
        config = Config(device="cpu", stream=True)
        source_config = SourceConfig()

        requirements = collect_install_requirements(config=config, source_config=source_config)

        self.assertFalse(requirements.include_sat)

    def test_isolation_requests_isolation_prefetch(self):
        config = Config(device="cpu")
        source_config = SourceConfig(isolate=True)

        requirements = collect_install_requirements(config=config, source_config=source_config)

        self.assertTrue(requirements.include_isolation)

    def test_apply_cache_env_replaces_hf_cache_paths_between_targets(self):
        with patch.dict(
            "os.environ",
            {
                "HF_HOME": "C:/old/hf",
                "HUGGINGFACE_HUB_CACHE": "C:/old/hf/hub",
                "XDG_CACHE_HOME": "C:/old/xdg",
            },
            clear=False,
        ):
            apply_cache_env("D:/models-a", offline=False)
            self.assertEqual(os.environ["HF_HOME"].replace("\\", "/"), "D:/models-a/hf")
            self.assertEqual(os.environ["HUGGINGFACE_HUB_CACHE"].replace("\\", "/"), "D:/models-a/hf/hub")
            self.assertEqual(os.environ["XDG_CACHE_HOME"].replace("\\", "/"), "D:/models-a/xdg")

            apply_cache_env("D:/models-b", offline=False)
            self.assertEqual(os.environ["HF_HOME"].replace("\\", "/"), "D:/models-b/hf")
            self.assertEqual(os.environ["HUGGINGFACE_HUB_CACHE"].replace("\\", "/"), "D:/models-b/hf/hub")
            self.assertEqual(os.environ["XDG_CACHE_HOME"].replace("\\", "/"), "D:/models-b/xdg")


if __name__ == "__main__":
    unittest.main()

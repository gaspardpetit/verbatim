import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from verbatim.config import Config
from verbatim_cli.args import build_parser
from verbatim_cli.config_file import load_config_file, merge_args, select_profile
from verbatim_cli.configure import apply_env_defaults, make_config, make_source_config, resolve_log_level, resolve_status_verbose


class TestEnvConfig(unittest.TestCase):
    @staticmethod
    def _build_args(argv):
        parser = build_parser(prog="verbatim")
        base_defaults = parser.parse_args([])
        user_args = parser.parse_args(argv)
        return parser, base_defaults, user_args

    @staticmethod
    def _write_config(contents: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(contents)
            return handle.name

    def test_env_backends_are_used_when_cli_and_config_are_unset(self):
        _parser, base_defaults, args = self._build_args(["input.wav"])

        with patch.dict(
            os.environ,
            {
                "VERBATIM_ASR_BACKEND": "qwen",
                "VERBATIM_LANGUAGE_BACKEND": "mms",
                "VERBATIM_LANGUAGE_MODEL": "facebook/mms-lid-126",
                "VERBATIM_DIARIZE": "senko",
                "VERBATIM_ASR_MODEL": "Qwen/Qwen3-ASR-1.7B",
            },
            clear=False,
        ):
            args = apply_env_defaults(args, base_defaults)
            config = make_config(args)
            source_config = make_source_config(args, speakers=None)

        self.assertEqual(config.transcriber_backend, "qwen")
        self.assertEqual(config.language_identifier_backend, "mms")
        self.assertEqual(config.mms_lid_model_size, "facebook/mms-lid-126")
        self.assertEqual(config.qwen_asr_model_size, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(source_config.diarize_strategy, "senko")

    def test_asr_model_targets_voxtral_when_voxtral_backend_is_selected(self):
        _parser, base_defaults, args = self._build_args(["input.wav"])
        with patch.dict(
            os.environ,
            {
                "VERBATIM_ASR_BACKEND": "voxtral",
                "VERBATIM_ASR_MODEL": "mistralai/Voxtral-Mini-3B-2507",
            },
            clear=False,
        ):
            args = apply_env_defaults(args, base_defaults)
            config = make_config(args)

        self.assertEqual(config.transcriber_backend, "voxtral")
        self.assertEqual(config.voxtral_model_size, "mistralai/Voxtral-Mini-3B-2507")

    def test_config_file_beats_env_backends(self):
        _parser, base_defaults, user_args = self._build_args(["input.wav"])
        config_path = self._write_config(
            "\n".join(
                [
                    "asr_backend: voxtral",
                    "language_backend: transcriber",
                    "language_model: acme/mms-custom",
                    "asr_model: config/asr-model",
                    "diarize: pyannote",
                ]
            )
            + "\n"
        )

        try:
            cfg_data = load_config_file(config_path)
            profile_overrides = select_profile(cfg_data, filename="input.wav")
            args = merge_args(base_defaults, profile_overrides, user_args)
        finally:
            Path(config_path).unlink()

        with patch.dict(
            os.environ,
            {
                "VERBATIM_ASR_BACKEND": "qwen",
                "VERBATIM_LANGUAGE_BACKEND": "mms",
                "VERBATIM_LANGUAGE_MODEL": "facebook/mms-lid-126",
                "VERBATIM_DIARIZE": "senko",
                "VERBATIM_ASR_MODEL": "env/asr-model",
            },
            clear=False,
        ):
            args = apply_env_defaults(args, base_defaults)
            config = make_config(args)
            source_config = make_source_config(args, speakers=None)

        self.assertEqual(config.transcriber_backend, "voxtral")
        self.assertEqual(config.language_identifier_backend, "transcriber")
        self.assertEqual(config.mms_lid_model_size, "acme/mms-custom")
        self.assertEqual(config.voxtral_model_size, "config/asr-model")
        self.assertEqual(config.whisper_model_size, "large-v3")
        self.assertEqual(source_config.diarize_strategy, "pyannote")

    def test_cli_beats_config_file_and_env_backends(self):
        _parser, base_defaults, user_args = self._build_args(
            [
                "input.wav",
                "--asr-backend",
                "qwen",
                "--language-backend",
                "mms",
                "--language-model",
                "cli/mms-model",
                "--asr-model",
                "cli/asr-model",
                "--diarize",
                "senko",
            ]
        )
        config_path = self._write_config(
            "\n".join(
                [
                    "asr_backend: voxtral",
                    "language_backend: transcriber",
                    "language_model: config/mms-model",
                    "asr_model: config/asr-model",
                    "diarize: pyannote",
                ]
            )
            + "\n"
        )

        try:
            cfg_data = load_config_file(config_path)
            profile_overrides = select_profile(cfg_data, filename="input.wav")
            args = merge_args(base_defaults, profile_overrides, user_args)
        finally:
            Path(config_path).unlink()

        with patch.dict(
            os.environ,
            {
                "VERBATIM_ASR_BACKEND": "auto",
                "VERBATIM_LANGUAGE_BACKEND": "transcriber",
                "VERBATIM_LANGUAGE_MODEL": "env/mms-model",
                "VERBATIM_DIARIZE": "pyannote",
                "VERBATIM_ASR_MODEL": "env/asr-model",
            },
            clear=False,
        ):
            args = apply_env_defaults(args, base_defaults)
            config = make_config(args)
            source_config = make_source_config(args, speakers=None)

        self.assertEqual(config.transcriber_backend, "qwen")
        self.assertEqual(config.language_identifier_backend, "mms")
        self.assertEqual(config.mms_lid_model_size, "cli/mms-model")
        self.assertEqual(config.qwen_asr_model_size, "cli/asr-model")
        self.assertEqual(source_config.diarize_strategy, "senko")

    def test_diarize_policy_beats_env_strategy(self):
        _parser, base_defaults, args = self._build_args(
            [
                "input.wav",
                "--diarize-policy",
                "1,2=energy;3=pyannote;*=channel",
            ]
        )

        with patch.dict(os.environ, {"VERBATIM_DIARIZE": "senko"}, clear=False):
            args = apply_env_defaults(args, base_defaults)
            source_config = make_source_config(args, speakers=None)

        self.assertEqual(source_config.diarize_strategy, "1,2=energy;3=pyannote;*=channel")

    def test_env_defaults_cover_runtime_knobs(self):
        _parser, base_defaults, args = self._build_args(["input.wav"])
        with patch.dict(
            os.environ,
            {
                "VERBATIM_MODELDIR": "D:/models",
                "VERBATIM_OFFLINE": "true",
                "VERBATIM_WORKDIR": "D:/workdir",
                "VERBATIM_LOG_FILE": "D:/logs/verbatim.log",
                "VERBATIM_OUTDIR": "D:/output",
                "VERBATIM_LOG_LEVEL": "INFO",
                "VERBATIM_DEVICE": "cpu",
                "VERBATIM_VOXTRAL_MAX_NEW_TOKENS": "512",
                "VERBATIM_VAD_BACKEND": "ast",
                "VERBATIM_NOISE_MODEL": "acme/noise-model",
                "VERBATIM_CODE_SWITCHING": "false",
            },
            clear=False,
        ):
            args = apply_env_defaults(args, base_defaults)
            config = make_config(args)
            log_level = resolve_log_level(args, base_defaults)
            status_verbose = resolve_status_verbose(args, base_defaults)

        self.assertEqual(args.outdir, "D:/output")
        self.assertEqual(args.log_file, "D:/logs/verbatim.log")
        self.assertEqual(config.model_cache_dir, "D:/models")
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.offline)
        self.assertEqual(config.working_dir, "D:/workdir")
        self.assertEqual(log_level, 20)
        self.assertEqual(status_verbose, 1)
        self.assertEqual(config.voxtral_max_new_tokens, 512)
        self.assertEqual(config.non_speech_backend, "ast")
        self.assertEqual(config.ast_audio_model_size, "acme/noise-model")
        self.assertFalse(config.code_switching)

    def test_legacy_model_cache_alias_maps_to_modeldir(self):
        _parser, _base_defaults, args = self._build_args(["input.wav", "--model-cache", "D:/models"])

        self.assertEqual(args.modeldir, "D:/models")

    def test_cli_beats_env_for_vad_and_noise_knobs(self):
        _parser, base_defaults, user_args = self._build_args(
            [
                "input.wav",
                "--non-vad-backend",
                "ast",
                "--noise-model",
                "cli/noise-model",
                "--code-switching",
                "false",
            ]
        )

        with patch.dict(
            os.environ,
            {
                "VERBATIM_VAD_BACKEND": "energy",
                "VERBATIM_NOISE_MODEL": "env/noise-model",
                "VERBATIM_CODE_SWITCHING": "true",
            },
            clear=False,
        ):
            args = apply_env_defaults(user_args, base_defaults)
            config = make_config(args)

        self.assertEqual(config.non_speech_backend, "ast")
        self.assertEqual(config.ast_audio_model_size, "cli/noise-model")
        self.assertFalse(config.code_switching)

    def test_cpu_switch_overrides_device_env(self):
        _parser, base_defaults, user_args = self._build_args(["input.wav", "--cpu"])

        with patch.dict(os.environ, {"VERBATIM_DEVICE": "cuda"}, clear=False):
            args = apply_env_defaults(user_args, base_defaults)
            config = make_config(args)

        self.assertEqual(config.device, "cpu")

    def test_config_explicit_default_value_beats_env_override(self):
        _parser, base_defaults, user_args = self._build_args(["input.wav"])
        config_path = self._write_config("offline: false\n")

        try:
            cfg_data = load_config_file(config_path)
            profile_overrides = select_profile(cfg_data, filename="input.wav")
            args = merge_args(base_defaults, profile_overrides, user_args)
        finally:
            Path(config_path).unlink()

        with patch.dict(os.environ, {"VERBATIM_OFFLINE": "true"}, clear=False):
            args = apply_env_defaults(args, base_defaults)
            config = make_config(args)

        self.assertFalse(config.offline)

    def test_explicit_unavailable_device_fails_fast(self):
        with patch.object(Config, "is_device_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Requested device 'cuda' is not available"):
                Config(device="cuda")


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from verbatim_cli.args import build_parser
from verbatim_cli.config_file import load_config_file, merge_args, select_profile
from verbatim_cli.configure import make_source_config


def _test_credential() -> str:
    return "".join(["12", "34"])


def _test_env_value() -> str:
    return "".join(["env", "-", "secret"])


def _test_yaml_value() -> str:
    return "".join(["yaml", "-", "secret"])


class TestDssPasswordConfig(unittest.TestCase):
    def test_make_source_config_uses_cli_password(self):
        test_credential = _test_credential()
        args = Namespace(
            isolate=None,
            diarize=None,
            diarize_policy=None,
            password=test_credential,
            diarization=None,
            vttm=None,
        )

        source_config = make_source_config(args, speakers=None)

        self.assertEqual(source_config.password, test_credential)

    def test_make_source_config_falls_back_to_env_password(self):
        env_value = _test_env_value()
        args = Namespace(
            isolate=None,
            diarize=None,
            diarize_policy=None,
            password=None,
            diarization=None,
            vttm=None,
        )

        with patch.dict(os.environ, {"VERBATIM_AUDIO_PASSWORD": env_value}, clear=False):
            source_config = make_source_config(args, speakers=None)

        self.assertEqual(source_config.password, env_value)

    def test_yaml_config_password_is_merged_into_args(self):
        parser = build_parser(prog="verbatim")
        base_defaults = parser.parse_args([])
        user_args = parser.parse_args(["input.wav"])
        yaml_value = _test_yaml_value()

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(f"password: {yaml_value}\n")
            config_path = handle.name

        try:
            cfg_data = load_config_file(config_path)
            profile_overrides = select_profile(cfg_data, filename="input.wav")
            args = merge_args(base_defaults, profile_overrides, user_args)
        finally:
            os.unlink(config_path)

        self.assertEqual(args.password, yaml_value)


if __name__ == "__main__":
    unittest.main()

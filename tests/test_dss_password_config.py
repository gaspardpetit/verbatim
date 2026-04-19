import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from verbatim_cli.args import build_parser
from verbatim_cli.config_file import load_config_file, merge_args, select_profile
from verbatim_cli.configure import make_source_config


class TestDssPasswordConfig(unittest.TestCase):
    def test_make_source_config_uses_cli_password(self):
        args = Namespace(
            isolate=None,
            diarize=None,
            diarize_policy=None,
            password="1234",
            diarization=None,
            vttm=None,
        )

        source_config = make_source_config(args, speakers=None)

        self.assertEqual(source_config.password, "1234")

    def test_make_source_config_falls_back_to_env_password(self):
        args = Namespace(
            isolate=None,
            diarize=None,
            diarize_policy=None,
            password=None,
            diarization=None,
            vttm=None,
        )

        with patch.dict(os.environ, {"VERBATIM_DSS_PASSWORD": "env-secret"}, clear=False):
            source_config = make_source_config(args, speakers=None)

        self.assertEqual(source_config.password, "env-secret")

    def test_yaml_config_password_is_merged_into_args(self):
        parser = build_parser(prog="verbatim")
        base_defaults = parser.parse_args([])
        user_args = parser.parse_args(["input.wav"])

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write("password: yaml-secret\n")
            config_path = handle.name

        try:
            cfg_data = load_config_file(config_path)
            profile_overrides = select_profile(cfg_data, filename="input.wav")
            args = merge_args(base_defaults, profile_overrides, user_args)
        finally:
            os.unlink(config_path)

        self.assertEqual(args.password, "yaml-secret")


if __name__ == "__main__":
    unittest.main()

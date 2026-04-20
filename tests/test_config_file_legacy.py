import unittest

from verbatim_cli.config_file import find_legacy_config_keys


class TestLegacyConfigWarnings(unittest.TestCase):
    def test_find_legacy_config_keys_detects_root_and_profile_entries(self):
        config_data = {
            "transcriber_backend": "qwen",
            "profiles": [
                {
                    "include": {"filename": ["*.wav"]},
                    "language_identifier_backend": "mms",
                    "mms_lid_model_size": "facebook/mms-lid-126",
                }
            ],
        }

        findings = find_legacy_config_keys(config_data)

        self.assertIn(("transcriber_backend", "asr_backend"), findings)
        self.assertIn(("profiles[0].language_identifier_backend", "language_backend"), findings)
        self.assertIn(("profiles[0].mms_lid_model_size", "language_model"), findings)


if __name__ == "__main__":
    unittest.main()

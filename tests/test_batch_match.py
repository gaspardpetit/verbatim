import tempfile
import unittest
from pathlib import Path

from verbatim_batch.main import build_batch_parser, collect_install_targets, filter_input_files, iter_input_files
from verbatim_cli.config_file import select_profile


class TestBatchMatch(unittest.TestCase):
    def test_iter_input_files_honors_custom_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "sample.dss").write_bytes(b"dss")
            (base / "sample.ds2").write_bytes(b"ds2")
            (base / "sample.wav").write_bytes(b"wav")

            inputs = iter_input_files(base, ["*.dss", "*.ds2"], recursive=False)

        self.assertEqual([path.name for path in inputs], ["sample.ds2", "sample.dss"])

    def test_filter_input_files_applies_ignore_patterns_by_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            kept = base / "keep.dss"
            skipped = base / "skip.ds2"
            kept.write_bytes(b"dss")
            skipped.write_bytes(b"ds2")

            filtered = filter_input_files([kept, skipped], ["skip.*"], batch_dir=base)

        self.assertEqual([path.name for path in filtered], ["keep.dss"])

    def test_filter_input_files_applies_ignore_patterns_by_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            nested_dir = base / "archive"
            nested_dir.mkdir()
            kept = base / "keep.ds2"
            skipped = nested_dir / "skip.dss"
            kept.write_bytes(b"ds2")
            skipped.write_bytes(b"dss")

            filtered = filter_input_files([kept, skipped], ["archive/*"], batch_dir=base)

        self.assertEqual([path.name for path in filtered], ["keep.ds2"])

    def test_collect_install_targets_uses_union_of_matched_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            wav_path = base / "sample.wav"
            mp3_path = base / "sample.mp3"
            wav_path.write_bytes(b"wav")
            mp3_path.write_bytes(b"mp3")

            cfg_data = {
                "batch_dir": str(base),
                "profiles": [
                    {"include": {"filename": ["*.wav"]}, "asr_backend": "qwen"},
                    {"include": {"filename": ["*.mp3"]}, "asr_backend": "voxtral", "diarize": "pyannote"},
                ],
            }
            parser = build_batch_parser()
            base_defaults = parser.parse_args([])
            user_args = parser.parse_args(["--batch-dir", str(base)])
            global_profile = select_profile(cfg_data, filename=None)

            targets = collect_install_targets(
                base_defaults=base_defaults,
                user_args=user_args,
                cfg_data=cfg_data,
                global_profile=global_profile,
                inputs=[wav_path, mp3_path],
            )

        backends = sorted(config.transcriber_backend for config, _source_config in targets)
        diarize_strategies = sorted(
            source_config.diarize_strategy for _config, source_config in targets if source_config.diarize_strategy is not None
        )

        self.assertIn("auto", backends)
        self.assertIn("qwen", backends)
        self.assertIn("voxtral", backends)
        self.assertEqual(["pyannote"], diarize_strategies)


if __name__ == "__main__":
    unittest.main()

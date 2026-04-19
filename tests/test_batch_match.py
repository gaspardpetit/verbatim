import tempfile
import unittest
from pathlib import Path

from verbatim_batch.main import filter_input_files, iter_input_files


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

            filtered = filter_input_files([kept, skipped], ["skip.*"])

        self.assertEqual([path.name for path in filtered], ["keep.dss"])


if __name__ == "__main__":
    unittest.main()

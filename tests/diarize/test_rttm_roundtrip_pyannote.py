import os
import tempfile
import unittest

from verbatim_files.rttm import load_rttm, write_rttm


class TestPyannoteRTTMInterop(unittest.TestCase):
    def test_pyannote_to_verbatim_rttm_roundtrip(self):
        # Raw RTTM lines that pyannote's load_rttm would parse
        rttm_lines = [
            "SPEAKER sample 1 0.000 1.000 <NA> <NA> SPK0 <NA> <NA>",
            "SPEAKER sample 1 1.000 0.500 <NA> <NA> SPK1 0.8 <NA>",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "source.rttm")
            dst = os.path.join(tmpdir, "roundtrip.rttm")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write("\n".join(rttm_lines) + "\n")

            annotation = load_rttm(src)
            write_rttm(annotation, dst)

            # Load again to confirm it stays parseable
            parsed = load_rttm(dst)

        self.assertEqual(len(parsed.segments), 2)
        self.assertEqual(parsed.segments[0].speaker, "SPK0")
        self.assertEqual(parsed.segments[1].speaker, "SPK1")
        self.assertAlmostEqual(parsed.segments[0].start, 0.0)
        self.assertAlmostEqual(parsed.segments[1].end, 1.5)


if __name__ == "__main__":
    unittest.main()

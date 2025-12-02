import os
import tempfile
import unittest

from verbatim_files.rttm import Annotation, Segment, load_rttm, loads_rttm, rttm_to_vttm, vttm_to_rttm, write_rttm
from verbatim_files.vttm import AudioRef, load_vttm, write_vttm


class TestRTTM(unittest.TestCase):
    def test_rttm_roundtrip_file(self):
        segments = [
            Segment(start=0.0, end=1.5, speaker="S1", file_id="sample", channel="1"),
            Segment(start=1.5, end=3.0, speaker="S2", file_id="sample", channel="1"),
        ]
        annotation = Annotation(segments=segments)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.rttm")
            write_rttm(annotation, path)
            parsed = load_rttm(path)

        self.assertEqual(len(parsed.segments), 2)
        self.assertEqual(parsed.segments[0].speaker, "S1")
        self.assertEqual(parsed.segments[1].speaker, "S2")
        self.assertAlmostEqual(parsed.segments[0].start, 0.0)
        self.assertAlmostEqual(parsed.segments[1].end, 3.0)

    def test_loads_rttm_from_string(self):
        rttm_text = "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> SPK0 <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> SPK1 0.9 <NA>",
            ]
        )
        annotation = loads_rttm(rttm_text)
        self.assertEqual(len(annotation.segments), 2)
        self.assertEqual(annotation.segments[0].speaker, "SPK0")
        self.assertIsNone(annotation.segments[0].confidence)
        self.assertAlmostEqual(annotation.segments[1].confidence or 0, 0.9, places=3)


class TestVTTM(unittest.TestCase):
    def test_vttm_roundtrip(self):
        annotation = Annotation(
            segments=[
                Segment(start=0.0, end=2.0, speaker="A", file_id="audio1"),
            ]
        )
        audio_refs = [AudioRef(id="audio1", path="/data/audio1.wav", channels=0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.vttm")
            write_vttm(path, audio=audio_refs, annotation=annotation)
            loaded_audio, loaded_annotation = load_vttm(path)

        self.assertEqual(len(loaded_audio), 1)
        self.assertEqual(loaded_audio[0].id, "audio1")
        self.assertEqual(loaded_audio[0].path, "/data/audio1.wav")
        self.assertEqual(len(loaded_annotation.segments), 1)
        self.assertEqual(loaded_annotation.segments[0].speaker, "A")

    def test_rttm_to_vttm_helper(self):
        annotation = Annotation(segments=[Segment(start=0.0, end=1.0, speaker="S1", file_id="clip")])
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            audio_refs = [AudioRef(id="clip", path=audio_path, channels=None)]
            rttm_path = os.path.join(tmpdir, "input.rttm")
            vttm_path = os.path.join(tmpdir, "converted.vttm")
            write_rttm(annotation, rttm_path)
            rttm_to_vttm(rttm_path, vttm_path, audio_refs=audio_refs)
            loaded_audio, loaded_annotation = load_vttm(vttm_path)
        self.assertEqual(loaded_audio[0].path, audio_path)
        self.assertEqual(len(loaded_annotation.segments), 1)

    def test_vttm_to_rttm_helper(self):
        annotation = Annotation(segments=[Segment(start=0.0, end=1.0, speaker="S1", file_id="clip")])
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            audio_refs = [AudioRef(id="clip", path=audio_path, channels=None)]
            vttm_path = os.path.join(tmpdir, "input.vttm")
            rttm_path = os.path.join(tmpdir, "converted.rttm")
            write_vttm(vttm_path, audio=audio_refs, annotation=annotation)
            vttm_to_rttm(vttm_path, rttm_path)
            parsed = load_rttm(rttm_path)
        self.assertEqual(len(parsed.segments), 1)

    def test_audio_ref_rejects_invalid_channel_spec(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            bad_path = tmp.name
            with self.assertRaises(ValueError):
                AudioRef(id="bad", path=bad_path, channels="")
            with self.assertRaises(ValueError):
                AudioRef(id="bad", path=bad_path, channels="*")
            with self.assertRaises(ValueError):
                AudioRef(id="bad", path=bad_path, channels="2-1")


if __name__ == "__main__":
    unittest.main()

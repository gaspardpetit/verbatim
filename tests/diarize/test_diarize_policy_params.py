import unittest

from verbatim.audio.sources.factory import relabel_speakers, resolve_clause_params
from verbatim_diarization.policy import PolicyClause
from verbatim_rttm import Segment


class TestDiarizePolicyParams(unittest.TestCase):
    def test_channel_params(self):
        clause = PolicyClause(targets={0}, strategy="channel", params={"speaker": "HOST", "offset": "1"})
        nb, kwargs = resolve_clause_params(clause, default_nb_speakers=3)
        self.assertEqual(nb, 3)
        self.assertEqual(kwargs.get("speaker"), "HOST")
        self.assertEqual(kwargs.get("offset"), 1)

    def test_speakers_override(self):
        clause = PolicyClause(targets={0}, strategy="energy", params={"speakers": "2"})
        nb, kwargs = resolve_clause_params(clause, default_nb_speakers=None)
        self.assertEqual(nb, 2)
        self.assertEqual(kwargs, {})

    def test_relabel_speakers(self):
        segments = [
            Segment(start=0, end=1, speaker="A", file_id="f"),
            Segment(start=1, end=2, speaker="B", file_id="f"),
        ]
        label_counts = {}
        relabeled = relabel_speakers(segments, base_label="HOST", label_counts=label_counts)
        self.assertEqual({s.speaker for s in relabeled}, {"HOST_1", "HOST_2"})
        relabeled2 = relabel_speakers([Segment(start=0, end=1, speaker="C", file_id="f")], base_label="HOST", label_counts=label_counts)
        self.assertEqual(relabeled2[0].speaker, "HOST_3")


if __name__ == "__main__":
    unittest.main()

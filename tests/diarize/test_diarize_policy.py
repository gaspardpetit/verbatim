import unittest

from verbatim_diarization.policy import assign_channels, parse_policy


class TestDiarizePolicy(unittest.TestCase):
    def test_parse_and_assign(self):
        policy_str = "1,2=energy;3=pyannote;*=channel?speaker=HOST"
        clauses = parse_policy(policy_str)
        self.assertEqual(len(clauses), 3)

        assignments = assign_channels(clauses, nchannels=4)
        # channel indices zero-based according to parser
        self.assertEqual(assignments[1].strategy, "energy")
        self.assertEqual(assignments[2].strategy, "energy")
        self.assertEqual(assignments[3].strategy, "pyannote")
        self.assertEqual(assignments[0].strategy, "channel")
        self.assertEqual(assignments[0].params.get("speaker"), "HOST")

    def test_range_and_default(self):
        policy_str = "0-2=channel;*=energy"
        clauses = parse_policy(policy_str)
        assignments = assign_channels(clauses, nchannels=5)
        for ch in (0, 1, 2):
            self.assertEqual(assignments[ch].strategy, "channel")
        for ch in (3, 4):
            self.assertEqual(assignments[ch].strategy, "energy")

    def test_no_wildcard(self):
        policy_str = "1=pyannote"
        clauses = parse_policy(policy_str)
        assignments = assign_channels(clauses, nchannels=2)
        self.assertEqual(assignments[1].strategy, "pyannote")
        self.assertNotIn(0, assignments)  # no wildcard means unassigned

    def test_speaker_pattern_params(self):
        policy_str = "0=channel?speaker=HOST&offset=1"
        clauses = parse_policy(policy_str)
        self.assertEqual(clauses[0].params.get("speaker"), "HOST")
        self.assertEqual(clauses[0].params.get("offset"), "1")


if __name__ == "__main__":
    unittest.main()

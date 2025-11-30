import unittest

from verbatim.eval.compare import compute_metrics_summary


class ComputeMetricsSummaryTests(unittest.TestCase):
    def test_gracefully_handles_missing_speaker_labels(self):
        """When only one side includes speaker labels, fall back to text-only metrics."""
        sample = {
            "utterances": [
                {
                    "utterance_id": "utt0",
                    "hyp_text": "hello world",
                    "hyp_spk": "",
                    "ref_text": "hello world",
                    "ref_spk": "1 1",
                }
            ]
        }

        metrics = compute_metrics_summary(sample)

        self.assertEqual(metrics.WER, 0.0)
        self.assertEqual(metrics.WDER, 0.0)
        self.assertEqual(metrics.cpWER, 0.0)


if __name__ == "__main__":
    unittest.main()

# Originally from https://github.com/google/speaker-id/blob/master/DiarizationLM/diarizationlm/utils_test.py
# SPDX-FileCopyrightText: 2025 Quan Wang and Yiling Huang and Guanlong Zhao and Evan Clark and Wei Xia and Hank Liao
# SPDX-License-Identifier: Apache-2.0

"""Test diarization evaluation utilities."""

import unittest
from typing import Any, Dict, cast

from verbatim.eval import diarization_utils as utils


class UtilsTest(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    def test_normalize_text(self):
        text = 'Hello,  HI, "how_are_you?" Good.'
        expected = "hello hi howareyou good"
        self.assertEqual(expected, utils.normalize_text(text))

    def test_speakers_transform(self):
        speakers = ["a", "b", "a", "d"]
        expected = ["1", "2", "1", "3"]
        self.assertListEqual(expected, utils.speakers_transform(speakers))

    def test_get_oracle_speakers(self):
        hyp_spk = "1 1 1 1 2 2 2 2"
        hyp_spk_align = "2 2 2 2 1 1 1 1"
        hyp_spk_oracle = utils.get_oracle_speakers(hyp_spk, hyp_spk_align)
        expected = [1, 1, 1, 1, 2, 2, 2, 2]
        self.assertEqual(expected, hyp_spk_oracle)

    def test_transcript_preserving_speaker_transfer(self):
        src_text = "hello good morning hi how are you pretty good"
        src_spk = "1 1 1 2 2 2 2 1 1"
        tgt_text = "hello morning hi hey are you be good"
        tgt_spk = "1 2 2 2 1 1 2 1"
        expected = "1 1 2 2 2 2 1 1"
        transferred_spk = utils.transcript_preserving_speaker_transfer(src_text, src_spk, tgt_text, tgt_spk)
        self.assertEqual(expected, transferred_spk)

    def test_ref_to_oracle(self):
        test_data = {
            "hyp_text": "yo hello hi wow great",
            "hyp_spk": "1 2 3 2 1",
            "ref_text": "hello hi hmm wow great",
            "ref_spk": "1 2 2 3 3",
        }
        self.assertEqual("1 2 3 1 1", utils.ref_to_oracle(test_data))

    def test_hyp_to_degraded(self):
        test_data = {
            "hyp_text": "yo hello hi wow great",
            "hyp_spk": "1 2 3 2 1",
            "ref_text": "hello hi hmm wow great",
            "ref_spk": "1 2 2 3 3",
        }
        self.assertEqual("1 2 2 1 3", utils.hyp_to_degraded(test_data))

    def test_create_diarized_text(self):
        word_labels = ["hi", "how", "are", "you", "good"]
        speaker_labels = ["1", "2", "2", "2", "1"]
        result = utils.create_diarized_text(word_labels, speaker_labels, use_new_line=False, speaker_prefix="<spk:", speaker_suffix=">")
        self.assertEqual("<spk:1> hi <spk:2> how are you <spk:1> good", result)

    def test_find_utt_dict(self):
        data_dict = {
            "utterances": [
                {
                    "utterance_id": "utt1",
                    "hyp_text": "how are you",
                },
                {
                    "utterance_id": "utt2",
                    "hyp_text": "good morning",
                },
            ]
        }
        result = utils.find_utt_dict("utt2", data_dict)
        if result is None:
            self.fail("find_utt_dict returned None for existing utterance id")
        res = cast(Dict[str, Any], result)
        self.assertEqual("good morning", res["hyp_text"])

    def test_update_hyp_text_in_utt_dict(self):
        utt_dict = {
            "utterance_id": "utt1",
            "hyp_text": "how are you good morning",
            "hyp_spk": "1 1 1 2 2",
        }

        new_hyp_text = "hello how are you good morning hi"
        updated = utils.update_hyp_text_in_utt_dict(utt_dict, new_hyp_text)
        self.assertEqual("1 1 1 1 2 2 2", updated["hyp_spk"])


if __name__ == "__main__":
    unittest.main()

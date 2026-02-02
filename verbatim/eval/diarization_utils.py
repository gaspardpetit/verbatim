# Originally from https://github.com/google/speaker-id/blob/master/DiarizationLM/diarizationlm/utils.py
# SPDX-FileCopyrightText: 2025 Quan Wang and Yiling Huang and Guanlong Zhao and Evan Clark and Wei Xia and Hank Liao
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import word_levenshtein as levenshtein
from scipy import optimize

PUNCTUATIONS = [",", ".", "_", "?", "!", "-", '"', "'"]


def normalize_text(text: str) -> str:
    """Normalize text."""
    # Convert to lower case.
    text_lower = text.lower().strip()

    # Remove punctuations.
    words = text_lower.split()
    new_words = []
    for word in words:
        new_word = word
        for punc in PUNCTUATIONS:
            replaced = new_word.replace(punc, "")
            if len(replaced.split()) != 1:
                continue
            new_word = replaced
        new_words.append(new_word)
    return " ".join(new_words)


def speakers_transform(speakers: Sequence[str]) -> list[str]:
    """Transform list of speakers to be order based."""
    spk_map = {}
    index = 0
    for spk in speakers:
        if spk not in spk_map:
            index += 1
            spk_map[spk] = index
    return [str(spk_map[spk]) for spk in speakers]


def get_aligned_hyp_speakers(
    hyp_text: str,
    ref_text: str,
    ref_spk: str,
    print_debug_info: bool = False,
) -> str:
    """Align ref_text to hyp_text, then apply the alignment to ref_spk."""
    # Counters for insertions and deletions in hyp and ref text alignment.
    num_insertions, num_deletions = 0, 0

    # Get the alignment.
    _, align = levenshtein.levenshtein_with_edits(normalize_text(ref_text), normalize_text(hyp_text))

    ref_spk_list = ref_spk.split()
    hyp_spk_align = []

    # Apply the alignment on ref speakers.
    for i, j in align:
        if i == -1:
            # hyp has insertion
            hyp_spk_align.append("-1")
            num_insertions += 1
        elif j == -1:
            # hyp has deletion
            num_deletions += 1
            continue
        else:
            hyp_spk_align.append(ref_spk_list[i])
    hyp_spk_align = " ".join(hyp_spk_align)

    if print_debug_info:
        print("Number of insertions: ", num_insertions)
        print("Number of deletions: ", num_deletions)
        # This is not the traditional denominator of WER. Instead, this is
        # len(hyp) + len(ref) - len(SUB).
        print("Length of align pairs: ", len(align))
    return hyp_spk_align


def get_oracle_speakers(hyp_spk: str, hyp_spk_align: str) -> Sequence[int]:
    """Get the oracle speakers for hypothesis."""
    hyp_spk_list = [int(x) for x in hyp_spk.split()]
    hyp_spk_align_list = [int(x) for x in hyp_spk_align.split()]

    # Build cost matrix.
    max_spk = max(*hyp_spk_list, *hyp_spk_align_list)
    cost_matrix = np.zeros((max_spk, max_spk))
    for aligned, original in zip(hyp_spk_align_list, hyp_spk_list):
        cost_matrix[aligned - 1, original - 1] += 1

    # Solve alignment.
    row_index, col_index = optimize.linear_sum_assignment(cost_matrix, maximize=True)

    # Build oracle.
    hyp_spk_oracle = hyp_spk_list.copy()
    for i, align_value in enumerate(hyp_spk_align_list):
        if align_value == -1:
            # There are some missing words. In such cases, we just use the original
            # speaker for these words if possible.
            if hyp_spk_list[i] == -1:
                # If we don't have original speaker for missing words, just use the
                # previous speaker if possible.
                # This is useful for the update_hyp_text_in_utt_dict() function.
                hyp_spk_oracle[i] = 1 if i == 0 else hyp_spk_oracle[i - 1]
            continue
        if row_index[align_value - 1] != align_value - 1:
            raise ValueError(f"Alignment mismatch at index {align_value - 1}: expected {align_value - 1}, got {row_index[align_value - 1]}")
        hyp_spk_oracle[i] = col_index[align_value - 1] + 1

    return hyp_spk_oracle


# Transcript-Preserving Speaker Transfer (TPST)
def transcript_preserving_speaker_transfer(src_text: str, src_spk: str, tgt_text: str, tgt_spk: str) -> str:
    """Apply source speakers to target."""
    if len(tgt_text.split()) != len(tgt_spk.split()):
        raise ValueError("tgt_text and tgt_spk must have the same length")
    if len(src_text.split()) != len(src_spk.split()):
        raise ValueError("src_text and src_spk must have the same length")
    tgt_spk_align = get_aligned_hyp_speakers(
        hyp_text=tgt_text,
        ref_text=src_text,
        ref_spk=src_spk,
    )
    oracle_speakers = get_oracle_speakers(hyp_spk=tgt_spk, hyp_spk_align=tgt_spk_align)
    return " ".join([str(x) for x in oracle_speakers])


def ref_to_oracle(json_dict: dict[str, str]) -> str:
    """Apply reference speakers to hypothesis."""
    return transcript_preserving_speaker_transfer(
        src_text=json_dict["ref_text"],
        src_spk=json_dict["ref_spk"],
        tgt_text=json_dict["hyp_text"],
        tgt_spk=json_dict["hyp_spk"],
    )


def hyp_to_degraded(json_dict: dict[str, str]) -> str:
    """Apply hypothesis speakers to reference."""
    return transcript_preserving_speaker_transfer(
        src_text=json_dict["hyp_text"],
        src_spk=json_dict["hyp_spk"],
        tgt_text=json_dict["ref_text"],
        tgt_spk=json_dict["ref_spk"],
    )


def create_diarized_text(
    word_labels: Sequence[str],
    speaker_labels: Sequence[str],
    use_new_line: bool = False,
    speaker_prefix: str = "<speaker:",
    speaker_suffix: str = ">",
) -> str:
    """Create diarized text from words and speaker labels."""
    output = []
    previous_speaker = None
    for word, speaker in zip(word_labels, speaker_labels):
        if speaker != previous_speaker:
            if previous_speaker and use_new_line:
                output.append("\n")
            output.append(speaker_prefix + speaker + speaker_suffix)
        output.append(word)
        previous_speaker = speaker
    return " ".join(output)


def find_utt_dict(utt_id: str, data_dict: dict[str, Any]) -> Optional[dict[str, str]]:
    """Find a utt_dict with a speicifc utterance_id from data_dict."""
    for utt_dict in data_dict["utterances"]:
        if utt_dict["utterance_id"] == utt_id:
            return utt_dict
    return None


def update_hyp_text_in_utt_dict(input_utt_dict: dict[str, str], new_hyp_text) -> dict[str, str]:
    """Update the hyp_text of a json utt_dict.

    We also transfer its original hyp_spk to the new hyp_text.

    This is useful if we want to use USM ASR transcripts to replace the
    turn-to-diarize transcripts, as the WER of turn-to-diarize transcripts is too
    high.

    Args:
      input_utt_dict: the input utt_dict
      new_hyp_text: the new hyp_text

    Returns:
      the new utt_dict
    """
    utt_dict = copy.deepcopy(input_utt_dict)
    # We don't know the speakers for new_hyp_text, so just use -1 as initial
    # speakers.
    new_hyp_spk = transcript_preserving_speaker_transfer(
        src_text=utt_dict["hyp_text"],
        src_spk=utt_dict["hyp_spk"],
        tgt_text=new_hyp_text,
        tgt_spk=" ".join(["-1" for _ in new_hyp_text.split()]),
    )
    # Update the utt_dict.
    utt_dict["hyp_text"] = new_hyp_text
    utt_dict["hyp_spk"] = new_hyp_spk
    utt_dict["hyp_diarized_text"] = create_diarized_text(new_hyp_text.split(), new_hyp_spk.split())
    return utt_dict

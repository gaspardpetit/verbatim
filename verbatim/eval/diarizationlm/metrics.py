# Originally from https://github.com/google/speaker-id/blob/master/DiarizationLM/diarizationlm/metrics.py
# SPDX-FileCopyrightText: 2025 Quan Wang and Yiling Huang and Guanlong Zhao and Evan Clark and Wei Xia and Hank Liao
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from typing import Any, Dict, Optional

import numpy as np
import tqdm
import word_levenshtein as levenshtein
from scipy import optimize

from . import utils

LOG = logging.getLogger(__name__)


def calculate_metrics(hyp_data: Dict, ref_data: Dict) -> Dict:
    """Calculate metrics between hypothesis and reference data."""
    data = {}

    # Validate speaker-text alignment
    for _i, utt in enumerate(ref_data["utterances"]):
        if len(utt["ref_text"].split()) != len(utt["ref_spk"].split()):
            raise ValueError(
                f"Utterance {utt['utterance_id']} has mismatched lengths:\n"
                f"Text ({len(utt['ref_text'].split())} words): {utt['ref_text']}\n"
                f"Speakers ({len(utt['ref_spk'].split())} labels): {utt['ref_spk']}"
            )

    # Squash everything into a single utterance
    data["utterances"] = [
        {
            "utterance_id": "utt0",
            "hyp_text": " ".join([utt["hyp_text"] for utt in hyp_data["utterances"]]),
            "hyp_spk": " ".join([utt["hyp_spk"] for utt in hyp_data["utterances"]]),
            "ref_text": " ".join([utt["ref_text"] for utt in ref_data["utterances"]]),
            "ref_spk": " ".join([utt["ref_spk"] for utt in ref_data["utterances"]]),
        }
    ]

    # Calculate metrics
    return compute_metrics_on_json_dict(data)


def format_metrics(metrics: Dict, prefix: str = "") -> str:
    """Format metrics as a string."""
    lines = [f"{prefix} Metrics:"]
    lines.extend(
        [
            f"  WER:        {metrics['WER']:.4f}",
            f"  WDER:       {metrics['WDER']:.4f}",
            f"  cpWER:      {metrics['cpWER']:.4f}",
            f"  SpkCntMAE:  {metrics['SpkCntMAE']:.4f}",
        ]
    )
    return "\n".join(lines)


def format_improvements(before_metrics: Dict, after_metrics: Dict) -> str:
    """Format improvements as a string with color coding."""
    lines = ["Improvements:"]
    for metric in ["WER", "WDER", "cpWER", "SpkCntMAE"]:
        improvement = before_metrics[metric] - after_metrics[metric]
        color_code = "\033[92m" if improvement > 0 else "\033[91m"
        reset_code = "\033[0m"
        lines.append(f"  {metric:<10}  {color_code}{improvement:.4f}{reset_code}")
    return "\n".join(lines)


@dataclasses.dataclass
class UtteranceMetrics:
    """Metrics for one utterance."""

    wer_insert: int = 0
    wer_delete: int = 0
    wer_sub: int = 0
    wer_correct: int = 0
    wer_total: int = 0

    wder_sub: int = 0
    wder_correct: int = 0
    wder_total: int = 0

    cpwer_insert: int = 0
    cpwer_delete: int = 0
    cpwer_sub: int = 0
    cpwer_correct: int = 0
    cpwer_total: int = 0

    speaker_count_error: int = 0


def merge_cpwer(wer_metrics: list[UtteranceMetrics], cpwer_metrics: UtteranceMetrics) -> None:
    """Compute cpWER metrics by merging a list of WER metrics."""
    for utt in wer_metrics:
        cpwer_metrics.cpwer_insert += utt.wer_insert
        cpwer_metrics.cpwer_delete += utt.wer_delete
        cpwer_metrics.cpwer_sub += utt.wer_sub
        cpwer_metrics.cpwer_correct += utt.wer_correct
        cpwer_metrics.cpwer_total += utt.wer_total


def compute_wer(hyp_text: str, ref_text: str) -> tuple[UtteranceMetrics, list[tuple[int, int]]]:
    """Compute the word error rate of an utterance."""
    result = UtteranceMetrics()
    hyp_normalized = utils.normalize_text(hyp_text)
    ref_normalized = utils.normalize_text(ref_text)
    hyp_words = hyp_normalized.split()
    ref_words = ref_normalized.split()

    # Get the alignment.
    _, align = levenshtein.levenshtein_with_edits(ref_normalized, hyp_normalized)

    # Apply the alignment on ref speakers.
    for i, j in align:
        if i == -1:
            result.wer_insert += 1
        elif j == -1:
            result.wer_delete += 1
        else:
            if ref_words[i] == hyp_words[j]:
                result.wer_correct += 1
            else:
                result.wer_sub += 1

    result.wer_total = result.wer_correct + result.wer_sub + result.wer_delete

    if result.wer_total != len(ref_words):
        raise ValueError(f"WER total mismatch: expected {len(ref_words)}, got {result.wer_total}. " f"Check the alignment and input normalization.")

    return result, align


def compute_utterance_metrics(
    hyp_text: str,
    ref_text: str,
    hyp_spk: Optional[str] = None,
    ref_spk: Optional[str] = None,
) -> UtteranceMetrics:
    """Compute all metrics of an utterance."""
    hyp_normalized = utils.normalize_text(hyp_text)
    ref_normalized = utils.normalize_text(ref_text)
    hyp_words = hyp_normalized.split()
    ref_words = ref_normalized.split()

    ########################################
    # Compute WER.
    ########################################
    result, align = compute_wer(hyp_text, ref_text)

    compute_diarization_metrics = hyp_spk or ref_spk
    if not compute_diarization_metrics:
        return result

    if not (hyp_spk and ref_spk):
        raise ValueError("hyp_spk and ref_spk must be both unset or both set.")

    ########################################
    # Compute WDER.
    ########################################
    hyp_spk_list = [int(x) for x in hyp_spk.split()]
    ref_spk_list = [int(x) for x in ref_spk.split()]
    if len(hyp_spk_list) != len(hyp_words):
        raise ValueError("hyp_spk and hyp_text must have the same length.")
    if len(ref_spk_list) != len(ref_words):
        raise ValueError("ref_spk and ref_text must have the same length.")
    hyp_spk_list_align = []
    ref_spk_list_align = []

    for i, j in align:
        if i != -1 and j != -1:
            ref_spk_list_align.append(ref_spk_list[i])
            hyp_spk_list_align.append(hyp_spk_list[j])

    # Build cost matrix.
    max_spk = max(*ref_spk_list_align, *hyp_spk_list_align)
    cost_matrix = np.zeros((max_spk, max_spk), dtype=int)
    for aligned, original in zip(ref_spk_list_align, hyp_spk_list_align):
        cost_matrix[aligned - 1, original - 1] += 1

    # Solve alignment.
    row_index, col_index = optimize.linear_sum_assignment(cost_matrix, maximize=True)
    result.wder_correct = int(cost_matrix[row_index, col_index].sum())
    result.wder_total = len(ref_spk_list_align)
    result.wder_sub = result.wder_total - result.wder_correct

    ########################################
    # Compute cpWER.
    ########################################
    spk_pair_metrics = {}
    cost_matrix = np.zeros((max_spk, max_spk), dtype=int)
    for i in range(1, max_spk + 1):
        ref_words_for_spk = [ref_words[k] for k in range(len(ref_words)) if ref_spk_list[k] == i]
        if not ref_words_for_spk:
            continue
        for j in range(1, max_spk + 1):
            hyp_words_for_spk = [hyp_words[k] for k in range(len(hyp_words)) if hyp_spk_list[k] == j]
            if not hyp_words_for_spk:
                continue
            spk_pair_metrics[(i, j)], _ = compute_wer(
                hyp_text=" ".join(hyp_words_for_spk),
                ref_text=" ".join(ref_words_for_spk),
            )
            cost_matrix[i - 1, j - 1] = spk_pair_metrics[(i, j)].wer_correct

    # Solve alignment.
    row_index, col_index = optimize.linear_sum_assignment(cost_matrix, maximize=True)
    metrics_to_concat = []
    for r, c in zip(row_index, col_index):
        if (r + 1, c + 1) not in spk_pair_metrics:
            continue
        metrics_to_concat.append(spk_pair_metrics[(r + 1, c + 1)])
    merge_cpwer(metrics_to_concat, result)

    ########################################
    # Compute speaker count error.
    ########################################
    hyp_spk_count = len(set(hyp_spk_list))
    ref_spk_count = len(set(ref_spk_list))
    result.speaker_count_error = hyp_spk_count - ref_spk_count

    return result


def compute_metrics_on_json_dict(
    json_dict: dict[str, Any],
    ref_text_field: str = "ref_text",
    hyp_text_field: str = "hyp_text",
    ref_spk_field: str = "ref_spk",
    hyp_spk_field: str = "hyp_spk",
) -> dict[str, Any]:
    """Compute metrics for all utterances in a json object."""
    compute_diarization_metrics = ref_spk_field or hyp_spk_field
    if compute_diarization_metrics:
        if not (ref_spk_field and hyp_spk_field):
            raise ValueError("hyp_spk_field and ref_spk_field must be both unset or both set.")
    result_dict = {
        "utterances": [],
    }
    for utt in tqdm.tqdm(json_dict["utterances"]):
        if compute_diarization_metrics:
            utt_metrics = compute_utterance_metrics(
                hyp_text=utt[hyp_text_field],
                ref_text=utt[ref_text_field],
                hyp_spk=utt[hyp_spk_field],
                ref_spk=utt[ref_spk_field],
            )
        else:
            utt_metrics = compute_utterance_metrics(
                hyp_text=utt[hyp_text_field],
                ref_text=utt[ref_text_field],
            )
        utt_result = dataclasses.asdict(utt_metrics)
        utt_result["utterance_id"] = utt["utterance_id"]
        result_dict["utterances"].append(utt_result)

    final_wer_total = 0
    final_wer_correct = 0
    final_wer_sub = 0
    final_wer_delete = 0
    final_wer_insert = 0
    final_wder_total = 0
    final_wder_correct = 0
    final_wder_sub = 0
    final_cpwer_total = 0
    final_cpwer_correct = 0
    final_cpwer_sub = 0
    final_cpwer_delete = 0
    final_cpwer_insert = 0
    final_speaker_count_absolute_error_total = 0
    for utt in result_dict["utterances"]:
        final_wer_total += utt["wer_total"]
        final_wer_correct += utt["wer_correct"]
        final_wer_sub += utt["wer_sub"]
        final_wer_delete += utt["wer_delete"]
        final_wer_insert += utt["wer_insert"]
        if compute_diarization_metrics:
            final_wder_total += utt["wder_total"]
            final_wder_correct += utt["wder_correct"]
            final_wder_sub += utt["wder_sub"]
            final_cpwer_total += utt["cpwer_total"]
            final_cpwer_correct += utt["cpwer_correct"]
            final_cpwer_sub += utt["cpwer_sub"]
            final_cpwer_delete += utt["cpwer_delete"]
            final_cpwer_insert += utt["cpwer_insert"]
            final_speaker_count_absolute_error_total += abs(utt["speaker_count_error"])

    result_dict["WER"] = (final_wer_sub + final_wer_delete + final_wer_insert) / final_wer_total

    if compute_diarization_metrics:
        result_dict["WDER"] = final_wder_sub / final_wder_total
        result_dict["cpWER"] = (final_cpwer_sub + final_cpwer_delete + final_cpwer_insert) / final_cpwer_total
        result_dict["SpkCntMAE"] = final_speaker_count_absolute_error_total / len(result_dict["utterances"])
    return result_dict

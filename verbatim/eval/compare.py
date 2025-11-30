import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, List

import tqdm

from ..transcript.words import Utterance
from .diarizationlm.metrics import UtteranceMetrics, compute_utterance_metrics


@dataclass
class UtteranceMetricsWithId(UtteranceMetrics):
    utterance_id: str = ""


@dataclass
class Metrics:
    # pylint: disable=invalid-name
    utterances: List[UtteranceMetricsWithId] = field(default_factory=list)
    WER: float = 0.0  # Word Error Rate - Measures transcription accuracy
    WDER: float = 0.0  # Word Diarization Error Rate - Measures speaker attribution errors
    cpWER: float = 0.0  # Concatenated-permutation WER - An enhanced WER that accounts for speaker permutation errors
    SpkCntMAE: float = 0.0  # Mean Absolute Error - Measures differences in estimated and actual speaker counts


def compute_metrics_summary(
    json_dict: dict[str, Any],
    ref_text_field: str = "ref_text",
    hyp_text_field: str = "hyp_text",
    ref_spk_field: str = "ref_spk",
    hyp_spk_field: str = "hyp_spk",
) -> Metrics:
    """Compute metrics for all utterances in a json object."""
    compute_diarization_metrics = bool(ref_spk_field) and bool(hyp_spk_field)
    if compute_diarization_metrics:
        for utt in json_dict.get("utterances", []):
            hyp_spk = utt.get(hyp_spk_field)
            ref_spk = utt.get(ref_spk_field)
            if not hyp_spk and not ref_spk:
                compute_diarization_metrics = False
                break
            if not (hyp_spk and ref_spk):
                LOG.warning("Speaker labels missing on one side; skipping diarization metrics for this comparison.")
                compute_diarization_metrics = False
                break
    result_dict = Metrics()
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
        utt_result = UtteranceMetricsWithId(**dataclasses.asdict(utt_metrics))
        utt_result.utterance_id = utt["utterance_id"]
        result_dict.utterances.append(utt_result)

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
    for utt in result_dict.utterances:
        final_wer_total += utt.wer_total
        final_wer_correct += utt.wer_correct
        final_wer_sub += utt.wer_sub
        final_wer_delete += utt.wer_delete
        final_wer_insert += utt.wer_insert
        if compute_diarization_metrics:
            final_wder_total += utt.wder_total
            final_wder_correct += utt.wder_correct
            final_wder_sub += utt.wder_sub
            final_cpwer_total += utt.cpwer_total
            final_cpwer_correct += utt.cpwer_correct
            final_cpwer_sub += utt.cpwer_sub
            final_cpwer_delete += utt.cpwer_delete
            final_cpwer_insert += utt.cpwer_insert
            final_speaker_count_absolute_error_total += abs(utt.speaker_count_error)

    result_dict.WER = (final_wer_sub + final_wer_delete + final_wer_insert) / final_wer_total

    if compute_diarization_metrics:
        result_dict.WDER = final_wder_sub / final_wder_total
        result_dict.cpWER = (final_cpwer_sub + final_cpwer_delete + final_cpwer_insert) / final_cpwer_total
        result_dict.SpkCntMAE = final_speaker_count_absolute_error_total / len(result_dict.utterances)
    return result_dict


LOG = logging.getLogger(__name__)


def compute_metrics(hyp_data: List[Utterance], ref_data: List[Utterance]) -> Metrics:
    """Calculate metrics between hypothesis and reference data."""
    data = {}

    def get_speaker_id(name: str, cache: List[str]):
        for idx, speaker in enumerate(cache):
            if speaker == name:
                return idx + 1  # Return a 1-indexed ID if the name exists.

        cache.append(name)
        return len(cache)  # The new speaker's ID is the new length of the cache.

    speakers = []
    ref_speakers = []
    data["utterances"] = [
        {
            "utterance_id": "utt0",
            "hyp_text": " ".join([utt.text for utt in hyp_data]),
            "hyp_spk": " ".join([f"{get_speaker_id(utt.speaker or 'None', speakers)} " * len(utt.text.split()) for utt in hyp_data]),
            "ref_text": " ".join([utt.text for utt in ref_data]),
            "ref_spk": " ".join([f"{get_speaker_id(utt.speaker or 'None', ref_speakers)} " * len(utt.text.split()) for utt in ref_data]),
        }
    ]

    metrics: Metrics = compute_metrics_summary(data)
    return metrics

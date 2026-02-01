import logging
import os
from typing import Optional

import numpy as np
import soundfile as sf

from verbatim.cache import get_required_cache
from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_diarization.utils import sanitize_uri_component
from verbatim_files.rttm import Annotation, Segment, dumps_rttm
from verbatim_files.vttm import AudioRef, dumps_vttm

LOG = logging.getLogger(__name__)


class EnergyDiarization(DiarizationStrategy):
    def __init__(self, energy_ratio_threshold: float = 1.18, normalize: bool = False):
        self.energy_ratio_threshold = energy_ratio_threshold
        self.normalize = normalize

    def _compute_channel_energies(self, audio: np.ndarray, start_sample: int, end_sample: int) -> tuple[float, float, float, float]:
        segment = audio[start_sample:end_sample]
        left = segment[:, 0]
        right = segment[:, 1]

        if self.normalize:
            # Normalize each channel independently to reduce bias from gain/DC offsets
            left = left - np.mean(left)
            right = right - np.mean(right)
            # Scale to +/-0.8 per channel to avoid over-amplification
            max_left = np.max(np.abs(left)) if left.size else 0.0
            max_right = np.max(np.abs(right)) if right.size else 0.0
            if max_left > 0:
                left = left * (0.8 / max_left)
            if max_right > 0:
                right = right * (0.8 / max_right)
            energy_left = np.sqrt(np.mean(left**2)) if left.size else 0.0
            energy_right = np.sqrt(np.mean(right**2)) if right.size else 0.0
        else:
            energy_left = float(np.sum(np.abs(left)))
            energy_right = float(np.sum(np.abs(right)))

        # Also compute peak energies for tiebreaking (use centered signals when normalized)
        peak_left = np.max(np.abs(left)) if left.size else 0.0
        peak_right = np.max(np.abs(right)) if right.size else 0.0

        return energy_left, energy_right, peak_left, peak_right

    def _determine_speaker(self, energy_left: float, energy_right: float, peak_left: float, peak_right: float) -> str:
        # If both channels carry effectively no energy, treat as silence
        epsilon = 1e-8
        if energy_left <= epsilon and energy_right <= epsilon:
            return "UNKNOWN"

        # Calculate energy ratios
        left_to_right_ratio = energy_left / energy_right if energy_right > 0 else float("inf")
        right_to_left_ratio = energy_right / energy_left if energy_left > 0 else float("inf")

        # Use primary energy ratio test
        if left_to_right_ratio > self.energy_ratio_threshold:
            return "SPEAKER_0"
        elif right_to_left_ratio > self.energy_ratio_threshold:
            return "SPEAKER_1"

        # If energy ratio is close (tie), use peak energy as tiebreaker
        # Simple direct comparison of peak values
        if peak_left > peak_right:
            return "SPEAKER_0"
        elif peak_right > peak_left:
            return "SPEAKER_1"

        return "UNKNOWN"

    def compute_diarization(self, file_path: str, out_rttm_file: Optional[str] = None, out_vttm_file: Optional[str] = None, **kwargs) -> Annotation:
        """
        Compute diarization based on stereo channel energy differences.
        When total energy difference is small, uses peak energy as a tiebreaker.

        Additional kwargs:
            segment_duration: Duration of analysis segments in seconds (default: 0.5)
        """
        audio, sample_rate = sf.read(file_path)
        audio_info = sf.info(file_path)
        LOG.info(f"Input file channels: {audio_info.channels}, sample rate: {audio_info.samplerate}")

        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("Stereo diarization requires stereo audio input")

        segment_duration = kwargs.get("segment_duration", 0.5)
        segment_samples = int(segment_duration * sample_rate)
        total_samples = len(audio)

        segments = []
        current_speaker = None
        segment_start = 0

        # Use the file name as the uri for the annotation
        uri = sanitize_uri_component(os.path.splitext(os.path.basename(file_path))[0])

        for start_sample in range(0, total_samples, segment_samples):
            end_sample = min(start_sample + segment_samples, total_samples)
            energy_left, energy_right, peak_left, peak_right = self._compute_channel_energies(audio, start_sample, end_sample)
            speaker = self._determine_speaker(energy_left, energy_right, peak_left, peak_right)

            if speaker not in (current_speaker, "UNKNOWN"):
                if current_speaker is not None:
                    segments.append(
                        Segment(
                            start=segment_start / sample_rate,
                            end=start_sample / sample_rate,
                            speaker=current_speaker,
                            file_id=uri,
                        )
                    )

                current_speaker = speaker
                segment_start = start_sample

        # Add the final segment
        if current_speaker is not None:
            segments.append(
                Segment(
                    start=segment_start / sample_rate,
                    end=total_samples / sample_rate,
                    speaker=current_speaker,
                    file_id=uri,
                )
            )

        annotation = Annotation(segments=segments, file_id=uri)

        if out_rttm_file:
            os.makedirs(os.path.dirname(out_rttm_file) or ".", exist_ok=True)
            get_required_cache().set_text(out_rttm_file, dumps_rttm(annotation))
            LOG.info("Wrote diarization to RTTM file: %s", out_rttm_file)

        if out_vttm_file:
            os.makedirs(os.path.dirname(out_vttm_file) or ".", exist_ok=True)
            audio_refs = [AudioRef(id=uri, path=file_path, channels=None)]
            get_required_cache().set_text(out_vttm_file, dumps_vttm(audio=audio_refs, annotation=annotation))
            LOG.info("Wrote diarization to VTTM file: %s", out_vttm_file)

        return annotation

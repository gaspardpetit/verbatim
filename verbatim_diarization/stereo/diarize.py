import logging
import os
from typing import Optional

import numpy as np
import soundfile as sf

from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_rttm import Annotation, AudioRef, Segment, write_vttm

LOG = logging.getLogger(__name__)


class StereoDiarization(DiarizationStrategy):
    def __init__(self, energy_ratio_threshold: float = 1.18):
        self.energy_ratio_threshold = energy_ratio_threshold

    def _compute_channel_energies(self, audio: np.ndarray, start_sample: int, end_sample: int) -> tuple[float, float, float, float]:
        segment = audio[start_sample:end_sample]
        energy_left = np.sum(np.abs(segment[:, 0]))
        energy_right = np.sum(np.abs(segment[:, 1]))

        # Also compute peak energies for tiebreaking
        peak_left = np.max(np.abs(segment[:, 0]))
        peak_right = np.max(np.abs(segment[:, 1]))

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
        uri = os.path.splitext(os.path.basename(file_path))[0]

        for start_sample in range(0, total_samples, segment_samples):
            end_sample = min(start_sample + segment_samples, total_samples)
            energy_left, energy_right, peak_left, peak_right = self._compute_channel_energies(audio, start_sample, end_sample)
            speaker = self._determine_speaker(energy_left, energy_right, peak_left, peak_right)

            if speaker != current_speaker and speaker != "UNKNOWN":
                if current_speaker is not None:
                    segment = Segment(segment_start / sample_rate, start_sample / sample_rate)
                    segments.append(Segment(start=segment.start, end=segment.end, speaker=current_speaker, file_id=uri))

                current_speaker = speaker
                segment_start = start_sample

        # Add the final segment
        if current_speaker is not None:
            segment = Segment(segment_start / sample_rate, total_samples / sample_rate)
            segments.append(Segment(start=segment.start, end=segment.end, speaker=current_speaker, file_id=uri))

        annotation = Annotation(segments=segments, file_id=uri)

        if out_rttm_file:
            os.makedirs(os.path.dirname(out_rttm_file) or ".", exist_ok=True)
            with open(out_rttm_file, "w", encoding="utf-8") as f:
                for segment, _track, label in annotation.itertracks(yield_label=True):  # pyright: ignore[reportAssignmentType]
                    f.write(f"SPEAKER {uri} 1 {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {label} <NA> <NA>\n")
            LOG.info("Wrote diarization to RTTM file: %s", out_rttm_file)

        if out_vttm_file:
            os.makedirs(os.path.dirname(out_vttm_file) or ".", exist_ok=True)
            audio_refs = [AudioRef(id=uri, path=file_path, channel="stereo")]
            write_vttm(out_vttm_file, audio=audio_refs, annotation=annotation)
            LOG.info("Wrote diarization to VTTM file: %s", out_vttm_file)

        return annotation

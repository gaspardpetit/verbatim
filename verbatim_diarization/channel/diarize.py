import logging
import os
from typing import Optional

import soundfile as sf

from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_rttm import Annotation, AudioRef, Segment, write_vttm

LOG = logging.getLogger(__name__)


class ChannelDiarization(DiarizationStrategy):
    """Treat each channel as a separate speaker without further diarization."""

    def __init__(self, speaker_labels: Optional[dict[int, str]] = None, speaker: str = "SPEAKER_{idx}", offset: int = 0):
        self.speaker_labels = speaker_labels or {}
        self.speaker_pattern = speaker
        self.speaker_offset = offset

    def compute_diarization(self, file_path: str, out_rttm_file: Optional[str] = None, out_vttm_file: Optional[str] = None, **kwargs) -> Annotation:
        audio, sample_rate = sf.read(file_path)
        audio_info = sf.info(file_path)
        LOG.info("Input file channels: %s, sample rate: %s", audio_info.channels, audio_info.samplerate)

        if audio.ndim == 1:
            # Expand mono to (samples, 1) so channel labeling still applies
            audio = audio[:, None]
        if audio.ndim < 2 or audio.shape[1] < 1:
            raise ValueError("Channel diarization requires audio with at least one channel")

        uri = os.path.splitext(os.path.basename(file_path))[0]
        duration = len(audio) / sample_rate

        segments = []
        audio_refs = []
        num_channels = audio.shape[1]
        for idx in range(num_channels):
            speaker = self.speaker_labels.get(idx, self.speaker_pattern.format(idx=idx + self.speaker_offset))
            segments.append(Segment(start=0.0, end=duration, speaker=speaker, file_id=uri))
            audio_refs.append(AudioRef(id=f"{uri}#{idx}", path=file_path, channels=str(idx)))

        annotation = Annotation(segments=segments, file_id=uri)

        if out_rttm_file:
            os.makedirs(os.path.dirname(out_rttm_file) or ".", exist_ok=True)
            with open(out_rttm_file, "w", encoding="utf-8") as f:
                for segment, _track, label in annotation.itertracks(yield_label=True):  # pyright: ignore[reportAssignmentType]
                    f.write(f"SPEAKER {uri} 1 {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {label} <NA> <NA>\n")
            LOG.info("Wrote channel diarization to RTTM file: %s", out_rttm_file)

        if out_vttm_file:
            os.makedirs(os.path.dirname(out_vttm_file) or ".", exist_ok=True)
            write_vttm(out_vttm_file, audio=audio_refs, annotation=annotation)
            LOG.info("Wrote channel diarization to VTTM file: %s", out_vttm_file)

        return annotation

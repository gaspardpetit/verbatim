import logging
import os
from typing import Optional

import soundfile as sf

from verbatim.cache import ArtifactCache
from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_diarization.utils import sanitize_uri_component
from verbatim_files.rttm import Annotation, Segment, dumps_rttm
from verbatim_files.vttm import AudioRef, dumps_vttm

LOG = logging.getLogger(__name__)


class ChannelDiarization(DiarizationStrategy):
    """Treat each channel as a separate speaker without further diarization."""

    def __init__(self, *, cache: ArtifactCache, speaker_labels: Optional[dict[int, str]] = None, speaker: str = "SPEAKER_{idx}", offset: int = 0):
        super().__init__(cache=cache)
        self.speaker_labels = speaker_labels or {}
        self.speaker_pattern = speaker
        self.speaker_offset = offset

    def compute_diarization(
        self,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_vttm_file: Optional[str] = None,
        status_hook=None,
        **kwargs,
    ) -> Annotation:
        _ = status_hook, kwargs
        buffer = self.cache.bytes_io(file_path)
        audio, sample_rate = sf.read(buffer)
        buffer.seek(0)
        audio_info = sf.info(buffer)
        LOG.info("Input file channels: %s, sample rate: %s", audio_info.channels, audio_info.samplerate)

        if audio.ndim == 1:
            # Expand mono to (samples, 1) so channel labeling still applies
            audio = audio[:, None]
        if audio.ndim < 2 or audio.shape[1] < 1:
            raise ValueError("Channel diarization requires audio with at least one channel")

        uri = sanitize_uri_component(os.path.splitext(os.path.basename(file_path))[0])
        duration = len(audio) / sample_rate

        segments = []
        audio_refs = []
        num_channels = audio.shape[1]
        for idx in range(num_channels):
            speaker = self.speaker_labels.get(idx, self.speaker_pattern.format(idx=idx + self.speaker_offset))
            segments.append(Segment(start=0.0, end=duration, speaker=speaker, file_id=uri))
            channel_id = sanitize_uri_component(f"{uri}_ch{idx}", fallback=f"{uri}_{idx}")
            audio_refs.append(AudioRef(id=channel_id, path=file_path, channels=str(idx)))

        annotation = Annotation(segments=segments, file_id=uri)

        if out_rttm_file:
            self.cache.set_text(out_rttm_file, dumps_rttm(annotation))
            LOG.info("Wrote channel diarization to RTTM file: %s", out_rttm_file)

        if out_vttm_file:
            self.cache.set_text(out_vttm_file, dumps_vttm(audio=audio_refs, annotation=annotation))
            LOG.info("Wrote channel diarization to VTTM file: %s", out_vttm_file)

        return annotation

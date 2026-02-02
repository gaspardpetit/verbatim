from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..audio import format_audio
from .audiosource import AudioSource, AudioStream

Annotation = object  # pylint: disable=invalid-name


def _to_float32(audio: NDArray) -> NDArray:
    if audio.dtype == np.float32:
        return audio
    if audio.dtype == np.int8:
        return audio.astype(np.float32) / 128.0
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    return audio.astype(np.float32)


class MemoryAudioStream(AudioStream):
    source: "MemoryAudioSource"

    def __init__(self, source: "MemoryAudioSource", diarization: Optional[Annotation]):
        super().__init__(start_offset=source.start_sample, diarization=diarization)
        self.source = source
        self._cursor = source.start_sample
        self.total_samples = source.total_samples
        self.end_sample = source.end_sample

    def next_chunk(self, chunk_length=1) -> NDArray:
        if not self.has_more():
            return np.array([], dtype=np.float32)

        chunk_samples = int(chunk_length * self.source.sample_rate)
        end_sample = min(self._cursor + chunk_samples, self.source.end_sample or self.source.total_samples)
        raw = self.source.samples[self._cursor : end_sample]
        self._cursor = end_sample

        if raw.size == 0:
            return np.array([], dtype=np.float32)

        if self.source.preserve_channels:
            return _to_float32(raw)

        # Mix down and resample to target rate via format_audio
        return format_audio(_to_float32(raw), from_sampling_rate=self.source.sample_rate)

    def has_more(self) -> bool:
        end = self.source.end_sample or self.source.total_samples
        return self._cursor < end

    def close(self):
        pass

    def get_nchannels(self) -> int:
        if self.source.samples.ndim == 1:
            return 1
        return self.source.samples.shape[1]

    def get_rate(self) -> int:
        return self.source.sample_rate


class MemoryAudioSource(AudioSource):
    samples: NDArray
    sample_rate: int
    diarization: Optional[Annotation]
    preserve_channels: bool
    start_sample: int
    end_sample: Optional[int]
    total_samples: int

    def __init__(
        self,
        *,
        samples: NDArray,
        sample_rate: int,
        diarization: Optional[Annotation] = None,
        preserve_channels: bool = False,
        channels_first: bool = False,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
        source_name: str = "<memory>",
    ):
        super().__init__(source_name=source_name)
        if channels_first and samples.ndim == 2:
            samples = samples.T
        self.samples = samples
        self.sample_rate = sample_rate
        self.diarization = diarization
        self.preserve_channels = preserve_channels
        self.start_sample = max(0, start_sample)
        self.end_sample = end_sample
        self.total_samples = samples.shape[0] if samples.ndim >= 1 else 0

    def open(self) -> MemoryAudioStream:
        return MemoryAudioStream(source=self, diarization=self.diarization)

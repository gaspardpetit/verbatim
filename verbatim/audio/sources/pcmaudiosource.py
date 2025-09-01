import logging
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from ..settings import AUDIO_PARAMS
from .audiosource import AudioSource, AudioStream

LOG = logging.getLogger(__name__)


class PCMInputStreamAudioStream(AudioStream):
    source: "PCMInputStreamAudioSource"
    _has_more: bool

    def __init__(self, source: "PCMInputStreamAudioSource"):
        super().__init__(start_offset=0, diarization=None)
        self.source = source
        self._has_more = True

    def next_chunk(self, chunk_length=1) -> NDArray:
        # Calculate the number of bytes needed per chunk
        bytes_per_sample = np.dtype(self.source.dtype).itemsize
        bytes_needed = chunk_length * self.source.sampling_rate * self.source.channels * bytes_per_sample

        # Buffer to store the read bytes
        input_data = bytearray()

        while len(input_data) < bytes_needed:
            chunk = self.source.stream.read(bytes_needed - len(input_data))
            if not chunk:
                # End of stream
                self._has_more = False
                break
            input_data.extend(chunk)

        # Convert the byte data to a NumPy array
        samples = np.frombuffer(input_data, dtype=self.source.dtype)
        return samples[: bytes_needed // bytes_per_sample]  # Ensure the array is the correct length

    def close(self):
        # caller is responsible for the lifecycle of the stream
        pass

    def has_more(self) -> bool:
        return self._has_more

    def get_nchannels(self) -> int:
        return self.source.channels

    def get_rate(self) -> int:
        return self.source.sampling_rate


class PCMInputStreamAudioSource(AudioSource):
    stream: BinaryIO
    channels: int
    sampling_rate: int
    dtype: np.dtype

    def __init__(
        self,
        *,
        source_name: str,
        stream: BinaryIO,
        channels: int = 1,
        sampling_rate: int = AUDIO_PARAMS.sample_rate,
        dtype: np.dtype = np.dtype(np.int16),
    ):
        super().__init__(source_name=source_name)
        self.stream = stream
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.dtype = dtype

    def open(self) -> PCMInputStreamAudioStream:
        return PCMInputStreamAudioStream(source=self)

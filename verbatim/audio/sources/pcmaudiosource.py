import logging
from typing import BinaryIO

import numpy as np

from .audiosource import AudioSource

LOG = logging.getLogger(__name__)

class PCMInputStreamAudioSource(AudioSource):
    stream: BinaryIO
    channels: int
    sampling_rate: int
    dtype: np.dtype
    _has_more: bool

    def __init__(self, stream: BinaryIO, channels: int = 1, sampling_rate: int = 16000, dtype=np.int16):
        super().__init__()
        self.stream = stream
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self._has_more = True

    def next_chunk(self, chunk_length=1) -> np.ndarray:
        # Calculate the number of bytes needed per chunk
        bytes_per_sample = np.dtype(self.dtype).itemsize
        bytes_needed = chunk_length * self.sampling_rate * self.channels * bytes_per_sample

        # Buffer to store the read bytes
        input_data = bytearray()

        while len(input_data) < bytes_needed:
            chunk = self.stream.read(bytes_needed - len(input_data))
            if not chunk:
                # End of stream
                self._has_more = False
                break
            input_data.extend(chunk)

        # Convert the byte data to a NumPy array
        samples = np.frombuffer(input_data, dtype=self.dtype)
        return samples[:bytes_needed // bytes_per_sample]  # Ensure the array is the correct length

    def open(self):
        # caller is responsible for the lifecycle of the stream
        pass

    def close(self):
        # caller is responsible for the lifecycle of the stream
        pass

    def has_more(self) -> bool:
        return self._has_more

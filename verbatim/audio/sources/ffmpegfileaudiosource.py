from typing import Optional
import logging

import numpy as np

# pylint: disable=c-extension-no-member
import av

from .audiosource import AudioSource, AudioStream
from ..audio import seconds_to_samples

LOG = logging.getLogger(__name__)

class PyAVAudioStream(AudioStream):
    source:"PyAVAudioSource"

    def __init__(self, source:"PyAVAudioSource"):
        """Open the container, find the audio stream, and seek if needed."""
        super().__init__(start_offset=seconds_to_samples(source.start_time), diarization=None)
        self.source = source

        # Internals
        self._container:av.container.InputContainer = None
        # pylint: disable=c-extension-no-member
        self._stream:av.audio.stream.AudioStream = None
        self._frame_iter = None

        # Buffer for leftover samples (when frames don't line up exactly with chunk size)
        self._sample_buffer = np.array([], dtype=np.float32)

        self._closed = False
        self._done_decoding = False


        LOG.info(f"Opening file with PyAV: {self.source.file_path}")
        self._container = av.open(self.source.file_path)

        # Find the first audio stream (or choose a specific one if needed)
        audio_streams = [s for s in self._container.streams if s.type == "audio"]
        if not audio_streams:
            raise ValueError("No audio streams found in file.")

        self._stream = audio_streams[0]
        self._stream.thread_type = "AUTO"  # allow FFmpeg to use threading if beneficial

        # If you want to force a certain sample format, channel layout, etc.,
        # you can set those on self._stream or handle them in code below.
        # E.g., self._stream.codec_context.sample_rate = self.target_sample_rate

        # If we want to seek to `start_time`, do so in seconds:
        if self.source.start_time > 0:
            # av.seek() or container.seek() uses timestamps in AV_TIME_BASE units
            # but PyAV usually accepts "seconds" if we specify 'any' for backward flag
            self._container.seek(
                int(self.source.start_time / self._stream.time_base),
                any_frame=False,
                stream=self._stream,
            )
            LOG.info(f"Seeking to {self.source.start_time} seconds.")

        # We create a generator that decodes frames from the audio stream
        # This is the raw frames from the container
        self._frame_iter = self._container.decode(self._stream)
        self._done_decoding = False
        self._closed = False

    def next_chunk(self, chunk_length=1) -> np.ndarray:
        """
        Return `chunk_length` seconds of audio as a NumPy array, shape (N,) or (N,1).
        If not enough data is available, decode more frames from PyAV.
        If at EOF, returns an empty array.
        """
        if self._closed:
            LOG.warning("next_chunk() called after close(). Returning empty array.")
            return np.array([], dtype=np.float32)

        if self._done_decoding and len(self._sample_buffer) == 0:
            # No more data left
            return np.array([], dtype=np.float32)

        # How many samples do we need?
        needed_samples = int(chunk_length * self.source.target_sample_rate)

        resampler = av.audio.resampler.AudioResampler(
            format="flt",  # float32
            layout="mono",  # 1 channel
            rate=16000,  # target samplerate
        )

        # Keep reading frames from PyAV until we have enough
        while len(self._sample_buffer) < needed_samples and not self._done_decoding:
            try:
                frame = next(self._frame_iter)
            except StopIteration:
                # No more frames from the container
                LOG.info("Reached end of audio stream.")
                self._done_decoding = True
                break

            # Optionally, we can check the frame's pts (presentation timestamp)
            # to see if we've passed end_time. If so, stop decoding.
            if self.source.end_time is not None and frame.pts is not None:
                # Convert pts to seconds
                # In PyAV, each stream has time_base -> frame_time = frame.pts * stream.time_base
                current_time_sec = float(frame.pts * self._stream.time_base)
                if current_time_sec > self.source.end_time:
                    LOG.info(
                        f"Reached end_time={self.source.end_time:.2f}s (current={current_time_sec:.2f}s). Stopping."
                    )
                    self._done_decoding = True
                    break

            # Resample to your desired format
            new_frames: list[av.audio.frame.AudioFrame] = resampler.resample(frame)
            for new_frame in new_frames:
                new_frame = (
                    new_frame.to_ndarray().astype(np.float32, copy=False).squeeze()
                )
                self._sample_buffer = np.concatenate([self._sample_buffer, new_frame])

        # By now, we either have enough samples or we hit EOF
        audio_array = self._sample_buffer[:needed_samples]
        self._sample_buffer = self._sample_buffer[needed_samples:]

        # For consistency, let's return a float32 numpy array shape (N,)
        audio_array = audio_array.astype(np.float32, copy=False)

        return audio_array

    def has_more(self) -> bool:
        """
        Return True if there's more data to read from the stream,
        or if there's still leftover samples in the buffer.
        """
        if self._closed:
            return False
        if not self._done_decoding:
            return True
        # If done decoding, but still have samples in buffer, it's True
        return len(self._sample_buffer) > 0

    def close(self):
        """Close the container and release resources."""
        if self._container and not self._closed:
            LOG.info("Closing PyAV container.")
            self._container.close()
        self._closed = True
        self._done_decoding = True
        self._sample_buffer = np.array([], dtype=np.float32)

class PyAVAudioSource(AudioSource):
    """
    A chunk-based audio reader that uses PyAV to decode audio frames.

    - Reads from a container (any format that FFmpeg/PyAV supports).
    - Decodes audio frames into an internal buffer.
    - Returns them in "chunk_length" second increments via next_chunk().
    - By default, it streams as 16-bit int PCM at some target rate (e.g. 16 kHz).
    """

    def __init__(self, file_path: str, target_sample_rate: int = 16000, start_time: float = 0.0, end_time: Optional[float] = None):
        """
        :param file_path: Path/URL to audio file
        :param target_sample_rate: Desired sample rate for output (e.g. 16k)
        :param start_time: Seek to this time (seconds) before reading
        :param end_time: Stop reading after this time (seconds) from start
        """
        super().__init__(source_name=file_path)
        self.file_path = file_path
        self.target_sample_rate = target_sample_rate
        self.start_time = start_time
        self.end_time = end_time

    def open(self):
        return PyAVAudioStream(source=self)

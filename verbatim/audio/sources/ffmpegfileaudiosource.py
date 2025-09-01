import logging
import math
from typing import Optional

# pylint: disable=c-extension-no-member
import av
import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from ..audio import seconds_to_samples
from .audiosource import AudioSource, AudioStream

LOG = logging.getLogger(__name__)


class PyAVAudioStream(AudioStream):
    source: "PyAVAudioSource"

    def __init__(self, source: "PyAVAudioSource"):
        """Open the container, find the audio stream, and seek if needed."""
        super().__init__(start_offset=seconds_to_samples(source.start_time), diarization=None)
        self.source = source

        # Internals
        # pylint: disable=c-extension-no-member
        self._container: av.container.InputContainer = None  # pyright: ignore[reportAttributeAccessIssue]
        self._stream: av.audio.stream.AudioStream = None  # pyright: ignore[reportAttributeAccessIssue]

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
        LOG.info(f"Audio stream channels: {self._stream.channels}")
        LOG.info(f"Preserve channels: {self.source.preserve_channels}")
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
        frame_iter = self._container.decode(self._stream)
        if frame_iter is None:
            raise RuntimeError("Frame iterator is not initialized. Decoding cannot proceed.")
        self._frame_iter = frame_iter

        self._done_decoding = False
        self._closed = False

    def next_chunk(self, chunk_length=1) -> NDArray:
        """
        Return `chunk_length` seconds of audio as a NumPy array.
        This version makes the resampling and reshaping behavior predictable and explicit.
        """

        if self._closed:
            LOG.warning("next_chunk() called after close(). Returning empty array.")
            return np.array([], dtype=np.float32)

        if self._done_decoding and len(self._sample_buffer) == 0:
            return np.array([], dtype=np.float32)

        # --- Step 1: Define original and target audio parameters ---
        original_sample_rate = self._stream.codec_context.sample_rate
        original_channels = self._stream.codec_context.channels

        target_sample_rate = self.source.target_sample_rate
        target_channels = original_channels if self.source.preserve_channels else 1

        LOG.debug(f"Original sample rate: {original_sample_rate} Hz, channels: {original_channels}")
        LOG.debug(f"Target sample rate: {target_sample_rate} Hz, channels: {target_channels}")

        # --- Step 2: Determine how many samples are needed for this chunk ---
        needed_target_samples = int(chunk_length * target_sample_rate)
        needed_input_samples = math.ceil(needed_target_samples * original_sample_rate / target_sample_rate)
        LOG.debug(f"Want {needed_target_samples} output samples → need ~{needed_input_samples} input samples")

        # --- Step 3: Decode and collect raw frames as-is ---
        collected = []

        total_collected_samples = 0
        while total_collected_samples < needed_input_samples and not self._done_decoding:
            try:
                frame = next(self._frame_iter)
            except StopIteration:
                LOG.info("Reached end of stream")
                self._done_decoding = True
                break

            # Respect end_time if set
            if self.source.end_time is not None and frame.pts is not None:
                timestamp_sec = float(frame.pts * self._stream.time_base)
                if timestamp_sec > self.source.end_time:
                    LOG.info(f"End time {self.source.end_time}s reached (current={timestamp_sec:.2f}s)")
                    self._done_decoding = True
                    break

            # Convert the frame to ndarray, float32
            arr = frame.to_ndarray().astype(np.float32, copy=False)

            # Normalize shape to always be (channels, samples)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]  # mono → (1, samples)
            elif arr.ndim == 2:
                pass  # already (channels, samples)
            else:
                raise ValueError(f"Unexpected frame shape: {arr.shape}")

            total_collected_samples += arr.shape[1]
            collected.append(arr)

        # --- Step 4: Stack collected frames into one array ---
        if not collected:
            return np.array([], dtype=np.float32)

        full_array = np.concatenate(collected, axis=1)  # shape: (channels, total_samples)
        LOG.debug(f"Concatenated shape before mixing: {full_array.shape}")

        # --- Optional: Downmix to mono if needed ---
        if target_channels == 1 and full_array.shape[0] > 1:
            # Average across channels: axis 0 is channels
            full_array = full_array.mean(axis=0, keepdims=True)  # shape becomes (1, total_samples)
            LOG.debug(f"Downmixed to mono: {full_array.shape}")

        # --- Step 5: Resample if needed ---
        if original_sample_rate != target_sample_rate:
            LOG.debug(f"Resampling from {original_sample_rate} → {target_sample_rate} Hz")

            # Resample each channel independently
            up = int(target_sample_rate)
            down = int(original_sample_rate)

            resampled_channels = []
            for ch_idx in range(full_array.shape[0]):
                resampled = resample_poly(full_array[ch_idx], up, down)
                resampled_channels.append(resampled)

            # Stack back into (channels, samples)
            full_array = np.stack(resampled_channels, axis=0)
            LOG.debug(f"Resampled shape: {full_array.shape}")
            LOG.debug(f"Chunk duration: {chunk_length} s → {needed_target_samples} samples needed at target rate")

        if target_channels == 1:
            return np.squeeze(full_array)
        else:
            return full_array.T

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

    def get_nchannels(self) -> int:
        return self._stream.channels

    def get_rate(self) -> int:
        return self.source.target_sample_rate


class PyAVAudioSource(AudioSource):
    """
    A chunk-based audio reader that uses PyAV to decode audio frames.

    - Reads from a container (any format that FFmpeg/PyAV supports).
    - Decodes audio frames into an internal buffer.
    - Returns them in "chunk_length" second increments via next_chunk().
    - By default, it streams as 16-bit int PCM at some target rate (e.g. 16 kHz).
    """

    def __init__(
        self,
        *,
        file_path: str,
        target_sample_rate: int = 16000,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        preserve_channels: bool = False,
    ):
        """
        :param file_path: Path/URL to audio file
        :param target_sample_rate: Desired sample rate for output (e.g. 16k)
        :param start_time: Seek to this time (seconds) before reading
        :param end_time: Stop reading after this time (seconds) from start
        :param preserve_channels: If True, return audio as-is (no channel mixing)
        """
        super().__init__(source_name=file_path)
        self.file_path = file_path
        self.target_sample_rate = target_sample_rate
        self.start_time = start_time
        self.end_time = end_time
        self.preserve_channels = preserve_channels

    def open(self):
        return PyAVAudioStream(source=self)

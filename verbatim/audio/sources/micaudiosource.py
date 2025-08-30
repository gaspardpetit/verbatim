import logging
import queue

import numpy as np
from numpy.typing import NDArray

import pyaudio
import sounddevice as sd

from .audiosource import AudioSource, AudioStream
from ..audio import samples_to_seconds

LOG = logging.getLogger(__name__)


class MicAudioStreamSoundDevice(AudioStream):
    source: "MicAudioSourceSoundDevice"
    audio_queue: queue.Queue
    stream:sd.InputStream

    def __init__(self, source: "MicAudioSourceSoundDevice"):
        super().__init__(start_offset=0, diarization=None)
        self.source = source
        self.audio_queue = queue.Queue()

        # Open the audio stream with the callback
        self.stream = sd.InputStream(
            samplerate=self.source.sampling_rate,
            channels=1,
            blocksize=self.source.frames_per_buffer,
            callback=self._audio_callback,
        )
        self.stream.start()
        LOG.info("Audio stream opened.")

    # pylint: disable=unused-argument
    def _audio_callback(self, indata, frames, time, status: sd.CallbackFlags):
        if status:
            LOG.warning(f"Audio stream status: {status}")
            if status.input_overflow:
                LOG.error("Input overflow occurred! Some audio data was lost.")
            if status.input_underflow:
                LOG.error("Input underflow occurred!")  # Add the audio data to the queue
        chunk = indata.copy().ravel()
        self.audio_queue.put(chunk)
        # LOG.debug(f"Captured new audio: len={len(chunk)} min={min(chunk)} max={max(chunk)}")

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            LOG.info("Audio stream closed.")

    def next_chunk(self, chunk_length=1) -> NDArray:
        """Fetch all available audio data from the queue without blocking."""
        frames = []
        while not self.audio_queue.empty():
            frames.append(self.audio_queue.get())

        if frames:
            # Concatenate all collected frames into a single numpy array
            audio_array = np.concatenate(frames, axis=0)
            # Convert the audio data to float32 and normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32)
            LOG.debug(f"Fetched {len(audio_array)} ({samples_to_seconds(len(audio_array))}) samples.")
            return audio_array

        else:
            # Return an empty array if no data is available
            return np.array([], dtype=np.float32)

    def has_more(self):
        return True

    def get_nchannels(self) -> int:
        stream:sd.InputStream = self.stream
        idevice, _odevice = stream.channels
        return idevice

    def get_rate(self) -> int:
        return self.stream._samplerate


class MicAudioSourceSoundDevice(AudioSource):
    def __init__(self, sampling_rate: int = 16000, frames_per_buffer: int = 1024):
        super().__init__(source_name="<mic>")
        self.sampling_rate = sampling_rate
        self.frames_per_buffer = frames_per_buffer

    def open(self):
        return MicAudioStreamSoundDevice(source=self)


class MicAudioStreamPyAudio(AudioStream):
    source: "MicAudioSourcePyAudio"
    p: pyaudio.PyAudio
    stream: pyaudio.Stream
    nchannels:int = 1

    def __init__(self, source: "MicAudioSourcePyAudio", nchannels:int = 1):
        super().__init__(start_offset=0, diarization=None)
        self.source = source
        self.p: pyaudio.PyAudio = pyaudio.PyAudio()
        self.stream: pyaudio.Stream = self.p.open(
            format=pyaudio.paInt16,
            channels=nchannels,
            rate=self.source.sampling_rate,
            input=True,
            frames_per_buffer=self.source.frames_per_buffer,
        )

    def next_chunk(self, chunk_length=1) -> NDArray:
        LOG.info(f"Recording {chunk_length} seconds of audio.")
        frames = []
        # Read exactly chunk_length seconds of audio
        for _ in range(
            0,
            int(self.source.frames_per_iter / self.source.frames_per_buffer * chunk_length),
        ):
            data = self.stream.read(self.source.frames_per_buffer)
            frames.append(data)

        audio_bytes = b"".join(frames)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        # Convert int16 array to float32 and normalize to [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32768.0
        LOG.info("Finished recording audio chunk.")
        return audio_array

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def has_more(self):
        return True

    def get_nchannels(self) -> int:
        return self.nchannels

    def get_rate(self) -> int:
        return self.source.sampling_rate


class MicAudioSourcePyAudio(AudioSource):
    frames_per_iter: int
    frames_per_buffer: int
    sampling_rate: int

    def __init__(
        self,
        latency: int = 16000,
        frames_per_buffer: int = 1000,
        sampling_rate: int = 16000,
    ):
        super().__init__(source_name="<mic>")
        self.frames_per_iter: int = latency
        self.frames_per_buffer: int = frames_per_buffer
        self.sampling_rate = sampling_rate

    def open(self) -> MicAudioStreamPyAudio:
        return MicAudioStreamPyAudio(source=self)

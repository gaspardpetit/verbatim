import io
import logging
import os
import wave
from typing import BinaryIO, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from verbatim.cache import ArtifactCache
from verbatim.voices.isolation import VoiceIsolation

from ..audio import format_audio, sample_to_timestr
from ..convert import convert_bytes_to_wav, convert_to_wav
from .audiosource import AudioSource, AudioStream

Annotation = object  # pylint: disable=invalid-name

LOG = logging.getLogger(__name__)

COMPATIBLE_FORMATS = [".mp3", ".m4a", ".wav"]


class FileAudioStream(AudioStream):
    source: "FileAudioSource"
    stream: wave.Wave_read

    def __init__(
        self,
        source: "FileAudioSource",
        diarization: Optional[Annotation],
        channel_indices: Optional[list[int]],
        file_id: Optional[str],
        cache: ArtifactCache,
    ):  # pylint: disable=too-many-positional-arguments
        super().__init__(start_offset=source.start_sample, diarization=diarization)
        self.source = source
        self.channel_indices = channel_indices
        self.file_id = file_id
        self._buffer: Optional[BinaryIO] = None
        if self.source.source_backend == "cache":
            input_bytes = cache.get_bytes(self.source.file_path)
            if not input_bytes:
                raise RuntimeError(
                    f"Cached bytes missing for input '{self.source.file_path}'. Populate the artifact cache before opening the audio source."
                )
            self._buffer = io.BytesIO(input_bytes)
        else:
            self._buffer = open(self.source.file_path, "rb")  # pylint: disable=consider-using-with
        self.stream = wave.open(self._buffer, "rb")
        total_frames = self.stream.getnframes()
        file_rate = self.stream.getframerate()
        total_seconds = total_frames / file_rate if file_rate else 0.0
        self.total_samples = int(round(total_seconds * 16000))
        self.end_sample = source.end_sample
        if self.source.start_sample != 0:
            self.setpos(self.source.start_sample)

    def setpos(self, new_sample_pos: int):
        file_samplerate = self.stream.getframerate()
        if file_samplerate != 16000:
            file_sample_pos = new_sample_pos * file_samplerate // 16000
        else:
            file_sample_pos = new_sample_pos
        self.stream.setpos(int(file_sample_pos))

    def next_chunk(self, chunk_length=1) -> NDArray:
        current_frame = self.stream.tell()
        LOG.debug(
            "Reading %s seconds of audio at %s from %s.",
            chunk_length,
            sample_to_timestr(current_frame, self.stream.getframerate()),
            self.source.file_path,
        )
        frames = self.stream.readframes(int(self.stream.getframerate() * chunk_length))
        sample_width = self.stream.getsampwidth()
        n_channels = self.stream.getnchannels()
        dtype = np.int16 if sample_width == 2 else np.int32 if sample_width == 4 else np.uint8
        audio_array = np.frombuffer(frames, dtype=dtype)
        audio_array = audio_array.reshape(-1, n_channels)

        if len(audio_array) == 0:
            return audio_array

        if self.channel_indices:
            try:
                audio_array = audio_array[:, self.channel_indices]
            except IndexError:
                LOG.warning("Requested channel indices %s exceed available channels %s", self.channel_indices, n_channels)
                return np.array([], dtype=np.float32)

        # Convert to float32
        audio_array = audio_array.astype(np.float32) / 32768.0

        if hasattr(self.source, "preserve_channels") and self.source.preserve_channels:
            # For diarization purposes, return stereo
            return audio_array
        else:
            # For transcription, convert to mono by averaging channels
            if audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)

            # Apply any additional formatting (resampling, etc.)
            audio_array = format_audio(audio_array, from_sampling_rate=self.stream.getframerate())

            return audio_array

    def has_more(self):
        current_frame = self.stream.tell()
        if self.source.end_sample is not None and current_frame > self.source.end_sample:
            return False
        total_frames = self.stream.getnframes()
        return current_frame < total_frames

    def close(self):
        self.stream.close()
        if self._buffer is not None:
            self._buffer.close()

    def get_nchannels(self) -> int:
        return self.stream.getnchannels()

    def get_rate(self) -> int:
        return self.stream.getframerate()


class FileAudioSource(AudioSource):
    diarization: Optional[Annotation]
    source_backend: Literal["cache", "path"]

    @staticmethod
    def _is_wave_readable_bytes(audio_bytes: bytes) -> bool:
        """Return True when the provided bytes are already a wave-readable WAV."""
        try:
            with io.BytesIO(audio_bytes) as buffer:
                with wave.open(buffer, "rb"):
                    return True
        except (wave.Error, OSError, EOFError):
            return False

    @staticmethod
    def _is_wave_readable_file(file_path: str) -> bool:
        """Return True when the on-disk file is already a wave-readable WAV."""
        try:
            with wave.open(file_path, "rb"):
                return True
        except (wave.Error, OSError, EOFError):
            return False

    @staticmethod
    def _normalized_wav_path(file_path_no_ext: str) -> str:
        """Return the artifact path used for normalized incompatible WAV inputs."""
        return file_path_no_ext + ".normalized.wav"

    def _require_cached_input_bytes(self) -> bytes:
        input_bytes = self.cache.get_bytes(self.file_path)
        if not input_bytes:
            raise RuntimeError(f"Cached bytes missing for input '{self.file_path}'. Populate the artifact cache before invoking the pipeline.")
        return input_bytes

    def _resolve_cache_backed_encoded_input(self, *, file_path_no_ext: str) -> str:
        """Resolve encoded cache-backed inputs to a PCM WAV artifact."""
        input_bytes = self._require_cached_input_bytes()
        return convert_bytes_to_wav(
            input_bytes=input_bytes,
            input_label=self.file_path,
            working_prefix_no_ext=file_path_no_ext,
            preserve_channels=self.preserve_channels,
            cache=self.cache,
        )

    def _resolve_path_backed_encoded_input(self, *, file_path_no_ext: str) -> str:
        """Resolve encoded path-backed inputs to a PCM WAV artifact."""
        return convert_to_wav(
            input_path=self.file_path,
            working_prefix_no_ext=file_path_no_ext,
            preserve_channels=self.preserve_channels,
            cache=self.cache,
        )

    def _resolve_cache_backed_wav_input(self, *, file_path_no_ext: str) -> str:
        """Resolve cache-backed WAV input to a wave-readable WAV artifact."""
        input_bytes = self._require_cached_input_bytes()
        if self._is_wave_readable_bytes(input_bytes):
            return self.file_path
        return convert_bytes_to_wav(
            input_bytes=input_bytes,
            input_label=self.file_path,
            working_prefix_no_ext=file_path_no_ext,
            output_path=self._normalized_wav_path(file_path_no_ext),
            preserve_channels=self.preserve_channels,
            cache=self.cache,
        )

    def _resolve_path_backed_wav_input(self, *, file_path_no_ext: str) -> str:
        """Resolve path-backed WAV input to a wave-readable WAV artifact."""
        if self._is_wave_readable_file(self.file_path):
            return self.file_path
        return convert_to_wav(
            input_path=self.file_path,
            working_prefix_no_ext=file_path_no_ext,
            output_path=self._normalized_wav_path(file_path_no_ext),
            preserve_channels=self.preserve_channels,
            cache=self.cache,
        )

    def _resolve_encoded_input(self, *, file_path_no_ext: str) -> str:
        """Resolve encoded input formats (mp3/m4a/...) to a PCM WAV artifact."""
        if self.source_backend == "cache":
            return self._resolve_cache_backed_encoded_input(file_path_no_ext=file_path_no_ext)
        if self.source_backend == "path":
            return self._resolve_path_backed_encoded_input(file_path_no_ext=file_path_no_ext)
        raise ValueError(f"Unsupported source backend: {self.source_backend}")

    def _resolve_wav_input(self, *, file_path_no_ext: str) -> str:
        """Resolve WAV input to a wave-readable PCM WAV artifact."""
        if self.source_backend == "cache":
            return self._resolve_cache_backed_wav_input(file_path_no_ext=file_path_no_ext)
        if self.source_backend == "path":
            return self._resolve_path_backed_wav_input(file_path_no_ext=file_path_no_ext)
        raise ValueError(f"Unsupported source backend: {self.source_backend}")

    def __init__(
        self,
        *,
        file: str,
        cache: ArtifactCache,
        diarization: Optional[Annotation],
        start_sample: int = 0,
        end_sample: Optional[int] = None,
        preserve_channels: bool = False,
        channel_indices: Optional[list[int]] = None,
        file_id: Optional[str] = None,
        source_backend: Literal["cache", "path"] = "cache",
    ):
        super().__init__(source_name=file)
        self.cache = cache
        self.file_path = file
        self.diarization = diarization
        self.channel_indices = channel_indices
        self.file_id = file_id
        self.preserve_channels = preserve_channels
        self.source_backend = source_backend
        file_path_no_ext, file_path_ext = os.path.splitext(self.file_path)
        file_path_ext = file_path_ext.lower()
        if file_path_ext == ".wav":
            self.file_path = self._resolve_wav_input(file_path_no_ext=file_path_no_ext)
        elif file_path_ext in COMPATIBLE_FORMATS:
            self.file_path = self._resolve_encoded_input(file_path_no_ext=file_path_no_ext)
        self.end_sample = end_sample
        self.start_sample = start_sample

    @staticmethod
    def isolate_voices(file_path: str, out_path_prefix: Optional[str] = None) -> Tuple[str, str]:
        LOG.info("Initializing Voice Isolation Model.")
        with VoiceIsolation(log_level=LOG.level) as voice_separator:
            if not out_path_prefix:
                basename, _ = os.path.splitext(os.path.basename(file_path))
                voice_prefix = f"{basename}-voice"
                noise_prefix = f"{basename}-noise"
            else:
                basename, ext = os.path.splitext(out_path_prefix)
                if ext:
                    voice_prefix = basename
                    noise_prefix = f"{basename}-noise"
                else:
                    voice_prefix = f"{basename}-voice"
                    noise_prefix = f"{basename}-noise"

            file_path, noise_path = voice_separator.isolate_voice_in_file(file=file_path, out_voice=voice_prefix, out_noise=noise_prefix)
        return file_path, noise_path

    def open(self):
        return FileAudioStream(
            source=self,
            diarization=self.diarization,
            channel_indices=self.channel_indices,
            file_id=self.file_id,
            cache=self.cache,
        )

import io
import wave

import numpy as np

from verbatim.cache import ArtifactCache

from .audiosource import AudioSource


class WavSink:
    @staticmethod
    def dump_to_wav(
        audio_source: AudioSource,
        output_path: str,
        *,
        cache: ArtifactCache,
        sample_rate: int = 16000,
        preserve_channels: bool = False,
    ):
        with audio_source.open() as audio_stream:
            input_sample_rate = audio_stream.get_rate()  # e.g., 48000
            num_channels = audio_stream.get_nchannels() if preserve_channels else 1
            sample_width = 2  # 16-bit PCM

            buffer = io.BytesIO()
            with wave.open(buffer, "w") as wav_file:
                # pylint: disable=no-member
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)

                while audio_stream.has_more():
                    audio_chunk: np.ndarray = np.asarray(audio_stream.next_chunk(chunk_length=1), dtype=np.float32)

                    # If stereo and not preserving channels, downmix to mono
                    if not preserve_channels and audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk.mean(axis=1)

                    # Resample the audio if needed
                    if input_sample_rate != sample_rate:
                        # Lazy import to avoid pulling scipy.signal during CLI startup
                        from scipy.signal import resample  # type: ignore  # pylint: disable=import-outside-toplevel

                        target_len = int(len(audio_chunk) * sample_rate / input_sample_rate)
                        audio_chunk = np.asarray(resample(audio_chunk, target_len))

                    # Convert to 16-bit PCM
                    int_samples = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16)
                    wav_file.writeframes(int_samples.tobytes())
            cache.set_bytes(output_path, buffer.getvalue())

import wave
import numpy as np

from .audiosource import AudioSource


class WavSink:
    @staticmethod
    def dump_to_wav(audio_source: AudioSource, output_path: str, sample_rate: int = 16000, preserve_channels: bool = False):
        """
        Dump the entire audio content from PyAVAudioSource to a .wav file.
        """
        # Open the PyAVAudioSource
        with audio_source.open() as audio_stream:
            # pylint: disable=no-member
            with wave.open(output_path, "w") as wav_file:
                # Set WAV parameters
                num_channels = audio_stream.get_nchannels() if preserve_channels else 1  # Stereo or mono
                sample_width = 2  # 16-bit PCM
                frame_rate = sample_rate  # Target sample rate
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(frame_rate)

                while audio_stream.has_more():
                    # Read a chunk of audio (default 1 second chunks)
                    audio_chunk = audio_stream.next_chunk(chunk_length=1)

                    # Convert float32 audio to 16-bit PCM for WAV format
                    int_samples = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16)

                    # Write to WAV file
                    wav_file.writeframes(int_samples.tobytes())

import logging
import os
from typing import Optional, Tuple

import librosa
import numpy as np
from audio_separator.separator import Separator
from numpy.typing import NDArray
from scipy.io.wavfile import write as wav_write

from ..audio.audio import format_audio

# Configure logger
LOG = logging.getLogger(__name__)


class VoiceIsolation:
    def __init__(self, log_level: int = logging.WARN, model_name: str = "MDX23C-8KFFT-InstVoc_HQ_2.ckpt"):
        self.separator = Separator(log_level=log_level, sample_rate=16000)
        cache_root = os.getenv("VERBATIM_MODEL_CACHE")
        offline_env = os.getenv("VERBATIM_OFFLINE", "0").lower() in ("1", "true", "yes")

        # If a cache is configured, prefer a cached checkpoint path under it
        model_path = model_name
        if cache_root and not os.path.isabs(model_name):
            candidate = os.path.join(cache_root, "audio-separator", model_name)
            if os.path.exists(candidate):
                model_path = candidate
            elif offline_env:
                raise RuntimeError(
                    f"Offline mode is enabled and isolation model '{model_name}' was not found at {candidate}"
                )

        self.separator.load_model(model_path)

    def __enter__(self) -> "VoiceIsolation":
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        del self.separator
        return False

    def isolate_voice_in_file(self, file: str, out_voice: Optional[str] = None, out_noise: Optional[str] = None) -> Tuple[str, str]:
        if out_voice is None:
            out_voice = "debug_audio-vocals"
        if out_noise is None:
            out_noise = "debug_audio-instrumental"

        # Use MDX to separate vocals from the source audio
        output_file_paths = self.separator.model_instance.separate(  # pyright: ignore[reportOptionalMemberAccess]
            file,
            {"Instrumental": out_noise, "Vocals": out_voice},
        )

        # The separated files: instrument track and vocal track
        instrument_audio_path = output_file_paths[0]
        voice_audio_path = output_file_paths[1]

        return voice_audio_path, instrument_audio_path

    def isolate_voice_in_array(self, audio: NDArray) -> NDArray:
        input_length = len(audio)

        if not np.any(audio):
            # audio is empty, skip
            return audio

        # Save the input audio to a temporary file
        temp_audio_file = "voice-isolation.wav"
        wav_write(temp_audio_file, 16000, audio)

        voice_audio_path, _ = self.isolate_voice_in_file(file=temp_audio_file)

        # Load the vocal audio back into a NumPy array
        voice_audio, voice_sampling_rate = librosa.load(voice_audio_path, sr=None, mono=False)
        voice_audio = voice_audio.T  # librosa formats (nchannel, samples) and we expect (samples, nchannels)

        # Format the vocal audio to mono and 16 kHz
        formatted_voice_audio = format_audio(voice_audio, int(voice_sampling_rate))
        wav_write("voice-isolation-filtered.wav", 16000, formatted_voice_audio)

        # ensure output has same length as input
        if len(formatted_voice_audio) < input_length:
            padding_length = input_length - len(formatted_voice_audio)
            formatted_voice_audio = np.pad(
                array=formatted_voice_audio,
                pad_width=(0, padding_length),
                mode="constant",
                constant_values=0,
            )
        elif len(formatted_voice_audio) > input_length:
            formatted_voice_audio = formatted_voice_audio[:input_length]

        return formatted_voice_audio

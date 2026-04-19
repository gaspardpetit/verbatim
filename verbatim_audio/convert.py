import io
import os
import wave
from typing import Optional, cast

import numpy as np

from verbatim.cache import ArtifactCache

from .audio import constrain_audio_range, resample_audio

DSS_EXTENSIONS = {".dss", ".ds2"}


def is_dss_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in DSS_EXTENSIONS


def _import_pydsscodec():
    # pylint: disable=import-outside-toplevel
    try:
        import pydsscodec
    except ImportError as exc:  # pragma: no cover - exercised via caller-facing error
        raise RuntimeError(
            "DSS/DS2 support requires the optional `pydsscodec` package. Install `pip install pydsscodec` or `verbatim[dss]`."
        ) from exc
    return pydsscodec


def _decoded_audio_to_wav_bytes(decoded_audio, *, output_sample_rate: int = 16000) -> bytes:
    source_sample_rate = int(getattr(decoded_audio, "sample_rate", 0) or getattr(decoded_audio, "native_rate", 0) or output_sample_rate)
    samples = np.asarray(decoded_audio.samples, dtype=np.float32)
    if samples.ndim != 1:
        samples = samples.reshape(-1)

    max_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
    # pydsscodec currently returns Python floats, but for DSS/DS2 samples these floats
    # can still be in PCM-scale units instead of normalized [-1, 1].
    if max_abs > 1.0:
        samples = samples / 32768.0

    if source_sample_rate != output_sample_rate:
        samples = resample_audio(samples, source_sample_rate, output_sample_rate)

    samples = constrain_audio_range(samples)
    pcm_samples = (samples * 32767.0).clip(-32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    wav_file = cast(wave.Wave_write, wave.open(buffer, "wb"))
    with wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(output_sample_rate)
        wav_file.writeframes(pcm_samples.tobytes())
    return buffer.getvalue()


def _decode_dss_file_to_wav_bytes(input_path: str, *, password: Optional[str] = None, output_sample_rate: int = 16000) -> bytes:
    pydsscodec = _import_pydsscodec()
    decoded_audio = pydsscodec.decode_file(input_path, password=password)
    return _decoded_audio_to_wav_bytes(decoded_audio, output_sample_rate=output_sample_rate)


def _decode_dss_bytes_to_wav_bytes(input_bytes: bytes, *, password: Optional[str] = None, output_sample_rate: int = 16000) -> bytes:
    pydsscodec = _import_pydsscodec()
    decoded_audio = pydsscodec.decode_bytes(input_bytes, password=password)
    return _decoded_audio_to_wav_bytes(decoded_audio, output_sample_rate=output_sample_rate)


def convert_to_wav(
    input_path: str,
    working_prefix_no_ext: str,
    preserve_channels: bool = False,
    overwrite=True,
    output_path: Optional[str] = None,
    *,
    password: Optional[str] = None,
    cache: ArtifactCache,
) -> str:
    # pylint: disable=import-outside-toplevel
    from .sources.ffmpegfileaudiosource import PyAVAudioSource
    from .sources.wavsink import WavSink

    converted_path = output_path or (working_prefix_no_ext + ".wav")
    if os.path.abspath(converted_path) == os.path.abspath(input_path):
        converted_path = working_prefix_no_ext + ".converted.wav"

    if not overwrite and os.path.exists(converted_path) is True:
        return converted_path

    if is_dss_path(input_path):
        cache.set_bytes(converted_path, _decode_dss_file_to_wav_bytes(input_path, password=password))
        return converted_path

    temp_file_audio_source = PyAVAudioSource(file_path=input_path, preserve_channels=preserve_channels)
    WavSink.dump_to_wav(
        audio_source=temp_file_audio_source,
        output_path=converted_path,
        preserve_channels=preserve_channels,
        cache=cache,
    )

    return converted_path


def convert_bytes_to_wav(
    *,
    input_bytes: bytes,
    input_label: str,
    working_prefix_no_ext: str,
    preserve_channels: bool = False,
    overwrite=True,
    output_path: Optional[str] = None,
    password: Optional[str] = None,
    cache: ArtifactCache,
) -> str:
    # pylint: disable=import-outside-toplevel
    from .sources.ffmpegfileaudiosource import PyAVAudioSource
    from .sources.wavsink import WavSink

    converted_path = output_path or (working_prefix_no_ext + ".wav")

    if not overwrite and cache.get_bytes(converted_path):
        return converted_path

    if is_dss_path(input_label):
        cache.set_bytes(converted_path, _decode_dss_bytes_to_wav_bytes(input_bytes, password=password))
        return converted_path

    temp_file_audio_source = PyAVAudioSource(
        file_path=input_label,
        file_obj=io.BytesIO(input_bytes),
        preserve_channels=preserve_channels,
    )
    WavSink.dump_to_wav(
        audio_source=temp_file_audio_source,
        output_path=converted_path,
        preserve_channels=preserve_channels,
        cache=cache,
    )

    return converted_path

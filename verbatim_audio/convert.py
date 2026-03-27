import io
import os
from typing import Optional

from verbatim.cache import ArtifactCache


def convert_to_wav(
    input_path: str,
    working_prefix_no_ext: str,
    preserve_channels: bool = False,
    overwrite=True,
    output_path: Optional[str] = None,
    *,
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
    cache: ArtifactCache,
) -> str:
    # pylint: disable=import-outside-toplevel
    from .sources.ffmpegfileaudiosource import PyAVAudioSource
    from .sources.wavsink import WavSink

    converted_path = output_path or (working_prefix_no_ext + ".wav")

    if not overwrite and cache.get_bytes(converted_path):
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

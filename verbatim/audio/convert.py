import os

def convert_to_wav(
        input_path: str,
        working_prefix_no_ext: str,
        preserve_channels: bool = False,
        overwrite = True) -> str:
    # pylint: disable=import-outside-toplevel
    from .sources.ffmpegfileaudiosource import PyAVAudioSource
    from .sources.wavsink import WavSink

    converted_path = working_prefix_no_ext + ".wav"

    if not overwrite and os.path.exists(converted_path) is True:
        return converted_path

    temp_file_audio_source = PyAVAudioSource(file_path=input_path, preserve_channels=preserve_channels)
    WavSink.dump_to_wav(audio_source=temp_file_audio_source, output_path=converted_path, preserve_channels=preserve_channels)

    return converted_path

import errno
import os
import sys
from typing import List, Optional, Union

import numpy as np
from pyannote.core.annotation import Annotation

from ...voices.diarize.factory import create_diarizer  # Add this import
from ..audio import samples_to_seconds, timestr_to_samples
from .audiosource import AudioSource
from .sourceconfig import SourceConfig


def convert_to_wav(input_path: str, working_prefix_no_ext: str, preserve_stereo: bool = False) -> str:
    # pylint: disable=import-outside-toplevel
    from .ffmpegfileaudiosource import PyAVAudioSource
    from .wavsink import WavSink

    temp_file_audio_source = PyAVAudioSource(file_path=input_path, preserve_channels=preserve_stereo)

    converted_path = working_prefix_no_ext + ".wav"
    WavSink.dump_to_wav(audio_source=temp_file_audio_source, output_path=converted_path, preserve_stereo=preserve_stereo)
    return converted_path


def compute_diarization(
    file_path: str,
    device: str,
    rttm_file: Optional[str] = None,
    strategy: str = "pyannote",
    nb_speakers: Union[int, None] = None,
) -> Annotation:
    """
    Compute diarization for an audio file using the specified strategy.

    Args:
        file_path: Path to audio file
        device: Device to use ('cpu' or 'cuda')
        rttm_file: Optional path to save RTTM file
        strategy: Diarization strategy ('pyannote' or 'stereo')
        nb_speakers: Optional number of speakers

        PyAnnote Annotation object
    """
    diarizer = create_diarizer(strategy=strategy, device=device, huggingface_token=os.getenv("HUGGINGFACE_TOKEN"))

    return diarizer.compute_diarization(file_path=file_path, out_rttm_file=rttm_file, nb_speakers=nb_speakers)


def create_audio_source(
    *,
    input_source: str,
    device: str,
    source_config: SourceConfig = SourceConfig(),
    start_time: Optional[str] = None,
    stop_time: Optional[str] = None,
    output_prefix_no_ext: str = "out",
    working_prefix_no_ext: str = "out",
    stream: bool = False,
) -> AudioSource:
    # pylint: disable=import-outside-toplevel

    if input_source == "-":
        from .pcmaudiosource import PCMInputStreamAudioSource

        return PCMInputStreamAudioSource(
            source_name="<stdin>",
            stream=sys.stdin.buffer,
            channels=1,
            sampling_rate=16000,
            dtype=np.dtype(np.int16),
        )

    elif input_source is None or input_source == ">":
        from .micaudiosource import MicAudioSourcePyAudio as MicAudioSource

        return MicAudioSource()

    start_sample: int = timestr_to_samples(start_time) if start_time else 0
    stop_sample: Optional[int] = timestr_to_samples(stop_time) if stop_time else None

    if os.path.exists(input_source) is False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_source)

    if source_config.diarization_file == "" or (source_config.diarize is not None and source_config.diarization_file is None):
        source_config.diarization_file = output_prefix_no_ext + ".rttm"

    from .ffmpegfileaudiosource import PyAVAudioSource
    from .fileaudiosource import FileAudioSource

    if os.path.splitext(input_source)[-1] != ".wav":
        if not (not stream and (source_config.isolate is not None or source_config.diarize is not None)):
            return PyAVAudioSource(
                file_path=input_source,
                start_time=samples_to_seconds(start_sample),
                end_time=samples_to_seconds(stop_sample) if stop_sample else None,
                preserve_channels=source_config.diarization_strategy == "stereo",
            )

        input_source = convert_to_wav(
            input_path=input_source,
            working_prefix_no_ext=working_prefix_no_ext,
            preserve_stereo=source_config.diarization_strategy == "stereo",
        )

        return create_audio_source(
            source_config=source_config,
            device=device,
            input_source=input_source,
            start_time=start_time,
            stop_time=stop_time,
            working_prefix_no_ext=working_prefix_no_ext,
            output_prefix_no_ext=output_prefix_no_ext,
        )

    if not stream:
        if source_config.isolate is not None:
            input_source, _noise_path = FileAudioSource.isolate_voices(file_path=input_source, out_path_prefix=working_prefix_no_ext)
        if source_config.diarize is not None:
            # Compute new diarization
            source_config.diarization = compute_diarization(
                file_path=input_source,
                device=device,
                rttm_file=source_config.diarization_file,
                strategy=source_config.diarization_strategy,
                nb_speakers=source_config.diarize,
            )
        elif source_config.diarization_file:
            # Load existing diarization from file
            from ...voices.diarization import Diarization

            try:
                source_config.diarization = Diarization.load_diarization(rttm_file=source_config.diarization_file)
            except (StopIteration, FileNotFoundError):
                # If the file doesn't exist or is empty, compute new diarization
                source_config.diarization = compute_diarization(
                    file_path=input_source,
                    device=device,
                    rttm_file=source_config.diarization_file,
                    strategy=source_config.diarization_strategy,
                    nb_speakers=source_config.diarize,
                )

    if source_config.diarization_file:
        from ...voices.diarization import Diarization

        source_config.diarization = Diarization.load_diarization(rttm_file=source_config.diarization_file)

    return FileAudioSource(
        file=input_source,
        start_sample=start_sample,
        end_sample=stop_sample,
        diarization=source_config.diarization,
        preserve_channels=source_config.diarization_strategy == "stereo",
    )


def create_separate_speaker_sources(
    *,
    input_source: str,
    device: str,
    source_config: SourceConfig = SourceConfig(),
    start_time: Optional[str] = None,
    stop_time: Optional[str] = None,
    output_prefix_no_ext: str = "out",
    working_prefix_no_ext: str = "out",
) -> List[AudioSource]:
    # pylint: disable=import-outside-toplevel

    if os.path.splitext(input_source)[-1] != ".wav":
        converted_input_source = convert_to_wav(input_path=input_source, working_prefix_no_ext=working_prefix_no_ext)
        return create_separate_speaker_sources(
            input_source=converted_input_source,
            device=device,
            source_config=source_config,
            start_time=start_time,
            stop_time=stop_time,
            output_prefix_no_ext=output_prefix_no_ext,
            working_prefix_no_ext=working_prefix_no_ext,
        )

    if source_config.diarization_file == "" or (source_config.diarize is not None and source_config.diarization_file is None):
        source_config.diarization_file = output_prefix_no_ext + ".rttm"

    nb_speakers = source_config.diarize
    if nb_speakers == 0:
        nb_speakers = None

    start_sample: int = timestr_to_samples(start_time) if start_time else 0
    stop_sample: Optional[int] = timestr_to_samples(stop_time) if stop_time else None

    from ...voices.separation import SpeakerSeparation
    from .fileaudiosource import FileAudioSource

    sources: List[AudioSource] = []

    with SpeakerSeparation(device=device, huggingface_token=os.getenv("HUGGINGFACE_TOKEN", "")) as separation:
        diarization, speaker_wav_files = separation.separate_speakers(
            file_path=input_source,
            out_rttm_file=source_config.diarization_file,
            out_speaker_wav_prefix=working_prefix_no_ext,
            nb_speakers=nb_speakers,
            diarization_strategy=source_config.diarization_strategy,
        )
        for _speaker, speaker_file in speaker_wav_files.items():
            sources.append(
                FileAudioSource(
                    file=speaker_file,
                    start_sample=start_sample,
                    end_sample=stop_sample,
                    diarization=diarization,
                )
            )

    return sources

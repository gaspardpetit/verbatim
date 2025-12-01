import errno
import logging
import os
import sys
import tempfile
from typing import Dict, List, Optional, Union

import numpy as np

from verbatim_diarization import create_diarizer  # Add this import
from verbatim_diarization.policy import assign_channels, parse_policy
from verbatim_diarization.separate import create_separator
from verbatim_rttm import Annotation as RTTMAnnotation
from verbatim_rttm import AudioRef, load_vttm, write_vttm

from ..audio import samples_to_seconds, timestr_to_samples
from ..convert import convert_to_wav
from .audiosource import AudioSource
from .sourceconfig import SourceConfig

LOG = logging.getLogger(__name__)

Annotation = RTTMAnnotation  # pylint: disable=invalid-name


def compute_diarization(
    file_path: str,
    device: str,
    *,
    rttm_file: Optional[str] = None,
    vttm_file: Optional[str] = None,
    strategy: str = "pyannote",
    nb_speakers: Union[int, None] = None,
    working_dir: Optional[str] = None,
) -> Annotation:
    """
    Compute diarization for an audio file using the specified strategy.

    Args:
        file_path: Path to audio file
        device: Device to use ('cpu' or 'cuda')
        rttm_file: Optional path to save RTTM file
        strategy: Diarization strategy ('pyannote', 'energy', or 'channel')
    nb_speakers: Optional number of speakers

        PyAnnote Annotation object
    """

    # If strategy string looks like a policy (contains assignment or ranges), apply policy routing
    if any(sym in strategy for sym in ("=", ";", ",", "-", "*")):
        return compute_diarization_policy(
            file_path=file_path,
            device=device,
            policy=strategy,
            rttm_file=rttm_file,
            vttm_file=vttm_file,
            nb_speakers=nb_speakers,
            working_dir=working_dir,
        )

    LOG.info(
        "Running diarization: strategy=%s nb_speakers=%s rttm_file=%s vttm_file=%s",
        strategy,
        nb_speakers,
        rttm_file,
        vttm_file,
    )
    diarizer = create_diarizer(strategy=strategy, device=device, huggingface_token=os.getenv("HUGGINGFACE_TOKEN"))

    return diarizer.compute_diarization(
        file_path=file_path,
        out_rttm_file=rttm_file,
        out_vttm_file=vttm_file,
        nb_speakers=nb_speakers,
        working_dir=working_dir,
    )


def _extract_channels(file_path: str, channels: List[int], working_dir: Optional[str]) -> str:
    # pylint: disable=import-outside-toplevel
    import soundfile as sf  # lazy import

    audio, sample_rate = sf.read(file_path)
    if isinstance(audio, np.ndarray) and audio.ndim > 1 and len(channels) > 0:
        subset = audio[:, channels]
    else:
        subset = audio
    tmp_dir = working_dir or tempfile.gettempdir()
    fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
    os.close(fd)
    sf.write(temp_path, subset, sample_rate)
    return temp_path


def compute_diarization_policy(
    *,
    file_path: str,
    device: str,
    policy: str,
    rttm_file: Optional[str],
    vttm_file: Optional[str],
    nb_speakers: Union[int, None],
    working_dir: Optional[str],
) -> Annotation:
    # pylint: disable=import-outside-toplevel
    import soundfile as sf  # lazy import

    clauses = parse_policy(policy)
    if len(clauses) == 0:
        return Annotation()

    info = sf.info(file_path)
    nchannels = info.channels
    assignments = assign_channels(clauses, nchannels=nchannels)

    if len(assignments) < nchannels:
        missing = sorted(set(range(nchannels)) - set(assignments.keys()))
        LOG.warning("Diarization policy left channels %s unassigned; they will be ignored.", missing)

    grouped: Dict[str, Dict] = {}
    for ch, clause in assignments.items():
        key = f"{clause.strategy}|{tuple(sorted(clause.params.items()))}"
        grouped.setdefault(key, {"clause": clause, "channels": set()})
        grouped[key]["channels"].add(ch)

    base_id = os.path.splitext(os.path.basename(file_path))[0]
    combined_segments = []
    temp_paths: List[str] = []

    try:
        for group in grouped.values():
            clause = group["clause"]
            channels = sorted(group["channels"])
            subset_path = file_path if len(channels) == nchannels else _extract_channels(file_path, channels, working_dir)
            if subset_path != file_path:
                temp_paths.append(subset_path)

            diarization = compute_diarization(
                file_path=subset_path,
                device=device,
                rttm_file=None,
                vttm_file=None,
                strategy=clause.strategy,
                nb_speakers=nb_speakers,
                working_dir=working_dir,
                **clause.params,
            )

            for segment in diarization.segments:
                segment.file_id = base_id
                combined_segments.append(segment)
    finally:
        for temp in temp_paths:
            # pylint: disable=broad-exception-caught
            try:
                os.unlink(temp)
            except Exception:  # pragma: no cover
                LOG.warning("Failed to remove temporary file %s", temp)

    merged = Annotation(segments=combined_segments, file_id=base_id)

    if rttm_file:
        with open(rttm_file, "w", encoding="utf-8") as f:
            merged.write_rttm(f)
    if vttm_file:
        write_vttm(vttm_file, audio=[AudioRef(id=base_id, path=file_path)], annotation=merged)

    return merged


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

    if source_config.diarization_file == "" or (source_config.diarize_strategy is not None and source_config.diarization_file is None):
        source_config.diarization_file = output_prefix_no_ext + ".rttm"
    if source_config.vttm_file is None:
        source_config.vttm_file = output_prefix_no_ext + ".vttm"
    if source_config.vttm_file == "":
        source_config.vttm_file = None
    working_dir = os.path.dirname(working_prefix_no_ext) or None

    from .ffmpegfileaudiosource import PyAVAudioSource
    from .fileaudiosource import FileAudioSource

    preserve_for_diarization = source_config.diarize_strategy is not None

    if os.path.splitext(input_source)[-1] != ".wav":
        if not (not stream and (source_config.isolate is not None or source_config.diarize_strategy is not None)):
            return PyAVAudioSource(
                file_path=input_source,
                start_time=samples_to_seconds(start_sample),
                end_time=samples_to_seconds(stop_sample) if stop_sample else None,
                preserve_channels=preserve_for_diarization,
            )

        input_source = convert_to_wav(
            input_path=input_source,
            working_prefix_no_ext=working_prefix_no_ext,
            preserve_channels=preserve_for_diarization,
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
        if source_config.vttm_file and not os.path.exists(source_config.vttm_file):
            LOG.info("No VTTM provided; creating minimal VTTM placeholder at %s", source_config.vttm_file)
            audio_id = os.path.splitext(os.path.basename(input_source))[0]
            preserve_channels = source_config.diarize_strategy in ("energy", "channel")
            audio_ref = AudioRef(id=audio_id, path=input_source, channel="stereo" if preserve_channels else "1")
            write_vttm(source_config.vttm_file, audio=[audio_ref], annotation=RTTMAnnotation())
        if source_config.vttm_file:
            try:
                _audio_refs, source_config.diarization = load_vttm(source_config.vttm_file)
                if len(source_config.diarization) == 0:
                    # Treat empty annotations as missing so diarization can be computed
                    source_config.diarization = None
            except (FileNotFoundError, ValueError):
                source_config.diarization = None

        if source_config.diarize_strategy is not None and source_config.diarization is None:
            # Compute new diarization
            nb_speakers = source_config.speakers if source_config.speakers not in (0, "") else None
            source_config.diarization = compute_diarization(
                file_path=input_source,
                device=device,
                rttm_file=source_config.diarization_file,
                vttm_file=source_config.vttm_file,
                strategy=source_config.diarize_strategy or "pyannote",
                nb_speakers=nb_speakers,
                working_dir=working_dir,
            )
        elif source_config.diarization is None:
            LOG.info("Diarization not requested; proceeding without diarization.")
        elif source_config.diarization_file and source_config.diarization is None:
            # Load existing diarization from file
            from verbatim_diarization import Diarization

            try:
                source_config.diarization = Diarization.load_diarization(rttm_file=source_config.diarization_file)
                if source_config.vttm_file and source_config.diarization is not None:
                    audio_id = os.path.splitext(os.path.basename(input_source))[0]
                    # Derive VTTM from RTTM for compatibility
                    write_vttm(source_config.vttm_file, audio=[AudioRef(id=audio_id, path=input_source)], annotation=source_config.diarization)
            except (StopIteration, FileNotFoundError):
                # If the file doesn't exist or is empty, compute new diarization
                nb_speakers = source_config.speakers if source_config.speakers not in (0, "") else None
                source_config.diarization = compute_diarization(
                    file_path=input_source,
                    device=device,
                    rttm_file=source_config.diarization_file,
                    vttm_file=source_config.vttm_file,
                    strategy=source_config.diarize_strategy or "pyannote",
                    nb_speakers=nb_speakers,
                    working_dir=working_dir,
                )

    if source_config.vttm_file and source_config.diarization is None:
        try:
            _audio_refs, source_config.diarization = load_vttm(source_config.vttm_file)
            if len(source_config.diarization) == 0:
                source_config.diarization = None
        except (FileNotFoundError, ValueError):
            source_config.diarization = None

    return FileAudioSource(
        file=input_source,
        start_sample=start_sample,
        end_sample=stop_sample,
        diarization=source_config.diarization,
        preserve_channels=False,
    )


def create_joint_speaker_sources(
    *,
    strategy: str = "pyannote",
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
        converted_input_source = convert_to_wav(input_path=input_source, working_prefix_no_ext=working_prefix_no_ext, preserve_channels=True)
        return create_joint_speaker_sources(
            input_source=converted_input_source,
            strategy=strategy,
            device=device,
            source_config=source_config,
            start_time=start_time,
            stop_time=stop_time,
            output_prefix_no_ext=output_prefix_no_ext,
            working_prefix_no_ext=working_prefix_no_ext,
        )

    if source_config.diarization_file == "" or (source_config.diarize_strategy is not None and source_config.diarization_file is None):
        source_config.diarization_file = output_prefix_no_ext + ".rttm"
    nb_speakers = source_config.speakers if source_config.speakers not in (0, "") else None

    start_sample: int = timestr_to_samples(start_time) if start_time else 0
    stop_sample: Optional[int] = timestr_to_samples(stop_time) if stop_time else None

    annotation: Annotation = compute_diarization(
        file_path=input_source,
        device=device,
        rttm_file=source_config.diarization_file,
        vttm_file=source_config.vttm_file,
        strategy=strategy,
        nb_speakers=nb_speakers,
    )

    from ..sources.fileaudiosource import FileAudioSource

    return [FileAudioSource(file=input_source, diarization=annotation, start_sample=start_sample, end_sample=stop_sample)]


def create_separate_speaker_sources(
    *,
    strategy: str = "pyannote",
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
        converted_input_source = convert_to_wav(input_path=input_source, working_prefix_no_ext=working_prefix_no_ext, preserve_channels=True)
        return create_separate_speaker_sources(
            input_source=converted_input_source,
            strategy=strategy,
            device=device,
            source_config=source_config,
            start_time=start_time,
            stop_time=stop_time,
            output_prefix_no_ext=output_prefix_no_ext,
            working_prefix_no_ext=working_prefix_no_ext,
        )

    if source_config.diarization_file == "" or (source_config.diarize_strategy is not None and source_config.diarization_file is None):
        source_config.diarization_file = output_prefix_no_ext + ".rttm"
    if source_config.vttm_file is None:
        source_config.vttm_file = output_prefix_no_ext + ".vttm"
    if source_config.vttm_file == "":
        source_config.vttm_file = None

    nb_speakers = source_config.speakers if source_config.speakers not in (0, "") else None

    start_sample: int = timestr_to_samples(start_time) if start_time else 0
    stop_sample: Optional[int] = timestr_to_samples(stop_time) if stop_time else None

    with create_separator(
        strategy=strategy,
        device=device,
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN", ""),
        diarization_strategy=source_config.diarize_strategy or "pyannote",
    ) as separation:
        sources = separation.separate_speakers(
            file_path=input_source,
            out_rttm_file=source_config.diarization_file,
            out_vttm_file=source_config.vttm_file,
            out_speaker_wav_prefix=working_prefix_no_ext,
            nb_speakers=nb_speakers,
            start_sample=start_sample,
            end_sample=stop_sample,
        )

    return sources

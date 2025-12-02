import errno
import logging
import os
import sys
import tempfile
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from verbatim_diarization import create_diarizer  # Add this import
from verbatim_diarization.policy import assign_channels, parse_params, parse_policy
from verbatim_diarization.separate import create_separator
from verbatim_files.rttm import Annotation as RTTMAnnotation
from verbatim_files.rttm import Segment, rttm_to_vttm
from verbatim_files.vttm import AudioRef, load_vttm, write_vttm

from ..audio import samples_to_seconds, timestr_to_samples
from ..convert import convert_to_wav
from .audiosource import AudioSource
from .sourceconfig import SourceConfig

LOG = logging.getLogger(__name__)

Annotation = RTTMAnnotation  # pylint: disable=invalid-name


def parse_channel_indices(channels_spec: Union[str, int, None]) -> List[int]:
    """Parse channel selections like '0', '0-2,4' into zero-based indices."""
    if channels_spec is None:
        return []
    if isinstance(channels_spec, str):
        spec_lower = channels_spec.strip().lower()
        if spec_lower in ("*", "stereo"):
            return []
    if isinstance(channels_spec, int):
        return [channels_spec]
    indices: Set[int] = set()
    for part in str(channels_spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            try:
                indices.add(int(part))
            except ValueError:
                continue
    return sorted(indices)


def filter_annotation_by_file_id(annotation: Optional[Annotation], file_id: Optional[str]) -> Optional[Annotation]:
    if annotation is None or not file_id:
        return annotation
    filtered_segments = [seg for seg in getattr(annotation, "segments", []) if getattr(seg, "file_id", None) in (file_id, "", None)]
    return Annotation(segments=filtered_segments, file_id=file_id)


def compute_diarization(
    file_path: str,
    device: str,
    *,
    rttm_file: Optional[str] = None,
    vttm_file: Optional[str] = None,
    strategy: str = "pyannote",
    nb_speakers: Union[int, None] = None,
    working_dir: Optional[str] = None,
    **strategy_kwargs,
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

    # Allow simple inline params: "energy?normalize=true"
    strategy_kwargs = dict(strategy_kwargs)
    if "?" in strategy and not any(sym in strategy for sym in ("=", ";", ",")):
        strategy, param_str = strategy.split("?", 1)
        strategy_kwargs.update(parse_params(param_str))

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
    if strategy == "energy" and "normalize" in strategy_kwargs:
        norm_val = strategy_kwargs.get("normalize")
        if isinstance(norm_val, str):
            norm_lower = norm_val.lower()
            strategy_kwargs["normalize"] = norm_lower in ("1", "true", "yes", "y")
        else:
            strategy_kwargs["normalize"] = bool(norm_val)
    diarizer = create_diarizer(strategy=strategy, device=device, huggingface_token=os.getenv("HUGGINGFACE_TOKEN"), **strategy_kwargs)

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


def resolve_clause_params(clause, default_nb_speakers: Union[int, None]) -> Tuple[Union[int, None], Dict[str, Union[str, int]], Optional[str]]:
    """Map policy params to diarizer kwargs and base speaker label (if provided)."""
    params_copy = dict(clause.params)
    nb_from_clause: Union[int, None]
    base_label = params_copy.get("speaker")
    if "speakers" in params_copy:
        try:
            nb_from_clause = int(params_copy.pop("speakers"))
        except ValueError:
            nb_from_clause = default_nb_speakers
    else:
        nb_from_clause = default_nb_speakers

    strategy_kwargs: Dict[str, Union[str, int]] = {}
    if clause.strategy == "channel":
        pattern = params_copy.pop("speaker", params_copy.pop("speaker_pattern", None)) or "SPEAKER_{idx}"
        offset = params_copy.pop("offset", 0) or 0
        try:
            offset_int = int(offset)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            offset_int = 0
        strategy_kwargs["speaker"] = pattern
        strategy_kwargs["offset"] = offset_int
    elif clause.strategy == "energy":
        normalize_val = params_copy.pop("normalize", None)
        if isinstance(normalize_val, str):
            normalize_lower = normalize_val.lower()
            if normalize_lower in ("1", "true", "yes", "y"):
                strategy_kwargs["normalize"] = True
            elif normalize_lower in ("0", "false", "no", "n"):
                strategy_kwargs["normalize"] = False
        elif normalize_val is not None:
            strategy_kwargs["normalize"] = bool(normalize_val)

    # Preserve any extra params for future strategies
    strategy_kwargs.update(params_copy)
    return nb_from_clause, strategy_kwargs, base_label


def relabel_speakers(segments: List[Segment], base_label: Optional[str], label_counts: Dict[str, int]) -> List[Segment]:
    """Rewrite speaker labels using a base_label. If multiple speakers, suffix with _# and dedupe globally."""
    if base_label is None:
        return segments

    seen_order = []
    for seg in segments:
        if seg.speaker not in seen_order:
            seen_order.append(seg.speaker)

    if len(seen_order) == 1:
        base = base_label
        count = label_counts.get(base, 0)
        if count == 0:
            new_label = base
        else:
            new_label = f"{base}_{count + 1}"
        label_counts[base] = count + 1
        mapping = {seen_order[0]: new_label}
    else:
        base = base_label
        count = label_counts.get(base, 0)
        mapping = {}
        for idx, speaker in enumerate(seen_order, start=1):
            new_label = f"{base}_{count + idx}"
            mapping[speaker] = new_label
        label_counts[base] = count + len(seen_order)

    new_segments = []
    for seg in segments:
        new_seg = Segment(
            start=seg.start,
            end=seg.end,
            speaker=mapping.get(seg.speaker, seg.speaker),
            file_id=seg.file_id,
            channel=seg.channel,
            orthography=seg.orthography,
            subtype=seg.subtype,
            confidence=seg.confidence,
            slat=seg.slat,
        )
        new_segments.append(new_seg)
    return new_segments


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
    label_counts: Dict[str, int] = {}
    audio_refs: List[AudioRef] = []

    try:
        for group in grouped.values():
            clause = group["clause"]
            channels = sorted(group["channels"])
            if info.channels < 2 and clause.strategy == "channel":
                LOG.warning("Channel diarization requested on mono input; skipping channel clause.")
                continue
            subset_path = file_path if len(channels) == nchannels else _extract_channels(file_path, channels, working_dir)
            if subset_path != file_path:
                temp_paths.append(subset_path)

            clause_nb_speakers, strategy_kwargs, base_label = resolve_clause_params(clause, nb_speakers)

            # Encode channel set as string for IDs and VTTM
            if channels:
                ranges = []
                start = prev = channels[0]
                for ch in channels[1:]:
                    if ch == prev + 1:
                        prev = ch
                        continue
                    ranges.append(str(start) if start == prev else f"{start}-{prev}")
                    start = prev = ch
                ranges.append(str(start) if start == prev else f"{start}-{prev}")
                channels_desc = ",".join(ranges)
            else:
                channels_desc = ""

            file_id = f"{base_id}#{channels_desc}" if channels_desc else base_id
            audio_refs.append(AudioRef(id=file_id, path=file_path, channels=channels_desc or "1"))

            diarization = compute_diarization(
                file_path=subset_path,
                device=device,
                rttm_file=None,
                vttm_file=None,
                strategy=clause.strategy,
                nb_speakers=clause_nb_speakers,
                working_dir=working_dir,
                **strategy_kwargs,
            )

            # Normalize diarizer output to a list of RTTM Segments
            if hasattr(diarization, "segments"):
                raw_segments = list(diarization.segments)  # verbatim_files.rttm.Annotation
            elif hasattr(diarization, "itertracks"):
                # pyannote-style Annotation
                raw_segments = [
                    Segment(start=seg.start, end=seg.end, speaker=str(label), file_id=file_id)
                    for seg, _track, label in diarization.itertracks(yield_label=True)  # type: ignore[attr-defined]
                ]
            else:
                raw_segments = []

            relabeled_segments = relabel_speakers(raw_segments, base_label, label_counts)
            for segment in relabeled_segments:
                segment.file_id = file_id
                combined_segments.append(segment)
    finally:
        for temp in temp_paths:
            # pylint: disable=broad-exception-caught
            try:
                os.unlink(temp)
            except Exception:  # pragma: no cover
                LOG.warning("Failed to remove temporary file %s", temp)

    merged = Annotation(segments=combined_segments, file_id=None)

    if rttm_file:
        with open(rttm_file, "w", encoding="utf-8") as f:
            merged.write_rttm(f)
    if vttm_file:
        write_vttm(vttm_file, audio=audio_refs, annotation=merged)

    # If all clauses were skipped/produced nothing, fall back to default strategy
    if len(merged) == 0:
        LOG.warning("Diarization policy produced no segments; falling back to strategy pyannote.")
        return compute_diarization(
            file_path=file_path,
            device=device,
            rttm_file=rttm_file,
            vttm_file=vttm_file,
            strategy="pyannote",
            nb_speakers=nb_speakers,
            working_dir=working_dir,
        )

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
    sources = create_audio_sources(
        input_source=input_source,
        device=device,
        source_config=source_config,
        start_time=start_time,
        stop_time=stop_time,
        output_prefix_no_ext=output_prefix_no_ext,
        working_prefix_no_ext=working_prefix_no_ext,
        stream=stream,
    )
    if not sources:
        raise RuntimeError("No audio sources could be created.")
    return sources[0]


def create_audio_sources(
    *,
    input_source: str,
    device: str,
    source_config: SourceConfig = SourceConfig(),
    start_time: Optional[str] = None,
    stop_time: Optional[str] = None,
    output_prefix_no_ext: str = "out",
    working_prefix_no_ext: str = "out",
    stream: bool = False,
) -> List[AudioSource]:
    # pylint: disable=import-outside-toplevel

    if input_source == "-":
        from .pcmaudiosource import PCMInputStreamAudioSource

        return [
            PCMInputStreamAudioSource(
                source_name="<stdin>",
                stream=sys.stdin.buffer,
                channels=1,
                sampling_rate=16000,
                dtype=np.dtype(np.int16),
            )
        ]

    elif input_source is None or input_source == ">":
        from .micaudiosource import MicAudioSourcePyAudio as MicAudioSource

        return [MicAudioSource()]

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

    preserve_for_diarization = source_config.diarize_strategy is not None or source_config.vttm_file is not None

    if os.path.splitext(input_source)[-1] != ".wav":
        if not (not stream and (source_config.isolate is not None or preserve_for_diarization)):
            return [
                PyAVAudioSource(
                    file_path=input_source,
                    start_time=samples_to_seconds(start_sample),
                    end_time=samples_to_seconds(stop_sample) if stop_sample else None,
                    preserve_channels=preserve_for_diarization,
                )
            ]

        input_source = convert_to_wav(
            input_path=input_source,
            working_prefix_no_ext=working_prefix_no_ext,
            preserve_channels=preserve_for_diarization,
        )

        return create_audio_sources(
            source_config=source_config,
            device=device,
            input_source=input_source,
            start_time=start_time,
            stop_time=stop_time,
            working_prefix_no_ext=working_prefix_no_ext,
            output_prefix_no_ext=output_prefix_no_ext,
        )

    audio_refs: List[AudioRef] = []

    if not stream:
        if source_config.isolate is not None:
            input_source, _noise_path = FileAudioSource.isolate_voices(file_path=input_source, out_path_prefix=working_prefix_no_ext)
        if source_config.vttm_file and not os.path.exists(source_config.vttm_file):
            LOG.info("No VTTM provided; creating minimal VTTM placeholder at %s", source_config.vttm_file)
            audio_id = os.path.splitext(os.path.basename(input_source))[0]
            audio_ref = AudioRef(id=audio_id, path=input_source, channels="*")
            write_vttm(source_config.vttm_file, audio=[audio_ref], annotation=RTTMAnnotation())
        if source_config.vttm_file:
            try:
                audio_refs, source_config.diarization = load_vttm(source_config.vttm_file)
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
                    rttm_to_vttm(
                        source_config.diarization_file,
                        source_config.vttm_file,
                        audio_refs=[AudioRef(id=audio_id, path=input_source, channels="1")],
                    )
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
            audio_refs, source_config.diarization = load_vttm(source_config.vttm_file)
            if len(source_config.diarization) == 0:
                source_config.diarization = None
        except (FileNotFoundError, ValueError):
            source_config.diarization = None

    # If VTTM defined multiple audio refs, emit one source per ref and scope diarization by file_id
    sources: List[AudioSource] = []
    if audio_refs:
        for audio_ref in audio_refs:
            channel_indices = parse_channel_indices(audio_ref.channels)
            file_path = audio_ref.path or input_source
            if not os.path.isabs(file_path):
                file_path = os.path.join(os.getcwd(), file_path)
            if not os.path.exists(file_path):
                LOG.warning("Audio path %s from VTTM not found; falling back to %s", file_path, input_source)
                file_path = input_source
            if os.path.splitext(file_path)[-1] != ".wav":
                # Ensure channel selection is preserved by converting to wav
                file_path = convert_to_wav(
                    input_path=file_path,
                    working_prefix_no_ext=f"{working_prefix_no_ext}-{audio_ref.id}",
                    preserve_channels=True,
                )
            diarization_obj = source_config.diarization if isinstance(source_config.diarization, RTTMAnnotation) else None
            scoped_diarization = filter_annotation_by_file_id(diarization_obj, audio_ref.id)
            sources.append(
                FileAudioSource(
                    file=file_path,
                    start_sample=start_sample,
                    end_sample=stop_sample,
                    diarization=scoped_diarization,
                    preserve_channels=False,
                    channel_indices=channel_indices or None,
                    file_id=audio_ref.id,
                )
            )
        return sources

    return [
        FileAudioSource(
            file=input_source,
            start_sample=start_sample,
            end_sample=stop_sample,
            diarization=source_config.diarization,
            preserve_channels=False,
        )
    ]


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

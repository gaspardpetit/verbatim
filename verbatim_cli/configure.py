"""Helpers to build configs and execution inputs from CLI args."""

import logging
import os
from argparse import Namespace
from typing import Callable, List, Optional, cast

from verbatim.config import Config
from verbatim_audio.sources.sourceconfig import SourceConfig
from verbatim_files.format.writer import TranscriptWriterConfig

LOG = logging.getLogger(__name__)


def _get_env_value(*env_names: str) -> Optional[str]:
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    return None


def _resolve_env_override(args, arg_name: str, *env_names: str) -> Optional[str]:
    value = getattr(args, arg_name, None)
    if value:
        return value
    return _get_env_value(*env_names)


def _parse_bool_env_value(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean environment value: {value}")


def _parse_int_env_value(value: str) -> int:
    return int(value.strip())


def _uses_default(args: Namespace, base_defaults: Namespace, arg_name: str) -> bool:
    explicit_args = getattr(args, "_explicit_args", set())
    if arg_name in explicit_args:
        return False
    if hasattr(args, "_get_kwargs"):
        current_value = getattr(args, arg_name, None)
        default_value = getattr(base_defaults, arg_name, None)
        if current_value == default_value:
            option_string = f"--{arg_name.replace('_', '-')}"
            if option_string in " ".join([]):
                return False
    return getattr(args, arg_name, None) == getattr(base_defaults, arg_name, None)


def _apply_string_env_default(args: Namespace, base_defaults: Namespace, arg_name: str, *env_names: str) -> None:
    if not _uses_default(args, base_defaults, arg_name):
        return
    env_value = _get_env_value(*env_names)
    if env_value:
        setattr(args, arg_name, env_value)


def _apply_bool_env_default(args: Namespace, base_defaults: Namespace, arg_name: str, *env_names: str) -> None:
    if not _uses_default(args, base_defaults, arg_name):
        return
    env_value = _get_env_value(*env_names)
    if env_value is None:
        return
    setattr(args, arg_name, _parse_bool_env_value(env_value))


def _apply_int_env_default(args: Namespace, base_defaults: Namespace, arg_name: str, *env_names: str) -> None:
    if not _uses_default(args, base_defaults, arg_name):
        return
    env_value = _get_env_value(*env_names)
    if env_value is None:
        return
    setattr(args, arg_name, _parse_int_env_value(env_value))


def apply_env_defaults(args: Namespace, base_defaults: Namespace) -> Namespace:
    for arg_name, env_name in (
        ("device", "VERBATIM_DEVICE"),
        ("modeldir", "VERBATIM_MODELDIR"),
        ("workdir", "VERBATIM_WORKDIR"),
        ("log_file", "VERBATIM_LOG_FILE"),
        ("outdir", "VERBATIM_OUTDIR"),
        ("asr_backend", "VERBATIM_ASR_BACKEND"),
        ("language_backend", "VERBATIM_LANGUAGE_BACKEND"),
        ("language_model", "VERBATIM_LANGUAGE_MODEL"),
        ("diarize", "VERBATIM_DIARIZE"),
        ("asr_model", "VERBATIM_ASR_MODEL"),
        ("vad_backend", "VERBATIM_VAD_BACKEND"),
        ("noise_model", "VERBATIM_NOISE_MODEL"),
    ):
        _apply_string_env_default(args, base_defaults, arg_name, env_name)

    if _uses_default(args, base_defaults, "password"):
        audio_password = _get_env_value("VERBATIM_AUDIO_PASSWORD")
        if audio_password:
            args.password = audio_password

    _apply_bool_env_default(args, base_defaults, "offline", "VERBATIM_OFFLINE")
    _apply_bool_env_default(args, base_defaults, "code_switching", "VERBATIM_CODE_SWITCHING")
    _apply_int_env_default(args, base_defaults, "voxtral_max_new_tokens", "VERBATIM_VOXTRAL_MAX_NEW_TOKENS")
    return args


def resolve_audio_password(args) -> Optional[str]:
    cli_password = getattr(args, "password", None)
    if cli_password:
        return cli_password
    return _get_env_value("VERBATIM_AUDIO_PASSWORD")


def resolve_asr_backend(args) -> Optional[str]:
    return _resolve_env_override(args, "asr_backend", "VERBATIM_ASR_BACKEND")


def resolve_language_backend(args) -> Optional[str]:
    return _resolve_env_override(args, "language_backend", "VERBATIM_LANGUAGE_BACKEND")


def resolve_language_model(args) -> Optional[str]:
    return _resolve_env_override(args, "language_model", "VERBATIM_LANGUAGE_MODEL")


def resolve_diarize_strategy(args) -> Optional[str]:
    diarize_strategy = getattr(args, "diarize_policy", None) or getattr(args, "diarize", None)
    if diarize_strategy:
        return diarize_strategy
    return _get_env_value("VERBATIM_DIARIZE")


def resolve_asr_model(args) -> Optional[str]:
    return _resolve_env_override(args, "asr_model", "VERBATIM_ASR_MODEL")


def resolve_device(args) -> str:
    if getattr(args, "cpu", False):
        return "cpu"
    return getattr(args, "device", "auto") or "auto"


def resolve_vad_backend(args) -> Optional[str]:
    return _resolve_env_override(args, "vad_backend", "VERBATIM_VAD_BACKEND")


def resolve_noise_model(args) -> Optional[str]:
    return _resolve_env_override(args, "noise_model", "VERBATIM_NOISE_MODEL")


def resolve_log_level(args: Namespace, base_defaults: Namespace) -> int:
    if not _uses_default(args, base_defaults, "verbose"):
        return compute_log_level(args.verbose)

    env_level = os.getenv("VERBATIM_LOG_LEVEL")
    if not env_level:
        return compute_log_level(args.verbose)

    try:
        return _parse_log_level(env_level)
    except ValueError:
        LOG.warning("Ignoring invalid VERBATIM_LOG_LEVEL=%s", env_level)
        return compute_log_level(args.verbose)


def resolve_status_verbose(args: Namespace, base_defaults: Namespace) -> int:
    if not _uses_default(args, base_defaults, "verbose"):
        return args.verbose

    env_level = os.getenv("VERBATIM_LOG_LEVEL")
    if not env_level:
        return args.verbose

    try:
        log_level = _parse_log_level(env_level)
    except ValueError:
        return args.verbose

    if log_level <= logging.DEBUG:
        return 2
    if log_level <= logging.INFO:
        return 1
    return 0


def _parse_log_level(value: str) -> int:
    normalized = value.strip().upper()
    aliases = {
        "WARN": "WARNING",
        "FATAL": "CRITICAL",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized.isdigit():
        return int(normalized)
    mapping = cast(Optional[Callable[[], dict[str, int]]], getattr(logging, "getLevelNamesMapping", None))
    if mapping is not None:
        level = mapping().get(normalized)
        if isinstance(level, int):
            return level
    legacy_level = logging.getLevelName(normalized)
    if isinstance(legacy_level, int):
        return legacy_level
    raise ValueError(f"Unknown log level: {value}")


def compute_log_level(verbose: int) -> int:
    if verbose <= 1:
        return logging.WARNING
    if verbose == 2:
        return logging.INFO
    return logging.DEBUG


def make_config(args) -> Config:
    config: Config = Config(
        device=resolve_device(args),
        output_dir=args.outdir,
        working_dir=args.workdir,
        stream=args.stream,
        offline=args.offline,
        model_cache_dir=args.modeldir,
        log_file=getattr(args, "log_file", None),
        log_colours=not bool(getattr(args, "log_file", None)),
    )
    asr_backend = resolve_asr_backend(args) or config.transcriber_backend
    asr_model = resolve_asr_model(args)
    if asr_model:
        normalized_backend = (asr_backend or "auto").lower()
        if normalized_backend in ("qwen", "qwen-asr"):
            config.qwen_asr_model_size = asr_model
        elif normalized_backend == "voxtral":
            config.voxtral_model_size = asr_model
        else:
            config.whisper_model_size = asr_model
    if getattr(args, "voxtral_max_new_tokens", None) is not None:
        config.voxtral_max_new_tokens = args.voxtral_max_new_tokens
    if asr_backend:
        config.transcriber_backend = asr_backend
    language_backend = resolve_language_backend(args)
    if language_backend:
        config.language_identifier_backend = language_backend
    if getattr(args, "language_detection_initial_seconds", None) is not None:
        config.language_detection_initial_seconds = args.language_detection_initial_seconds
    if getattr(args, "language_detection_increment_seconds", None) is not None:
        config.language_detection_increment_seconds = args.language_detection_increment_seconds
    if getattr(args, "language_detection_factor", None) is not None:
        config.language_detection_factor = args.language_detection_factor
    language_model = resolve_language_model(args)
    if language_model:
        config.mms_lid_model_size = language_model
    vad_backend = resolve_vad_backend(args)
    if vad_backend:
        config.non_speech_backend = vad_backend
    noise_model = resolve_noise_model(args)
    if noise_model:
        config.ast_audio_model_size = noise_model
    if getattr(args, "code_switching", None) is not None:
        config.code_switching = args.code_switching
    config.lang = args.languages if args.languages else ["en"]
    return config


def make_write_config(args, log_level: int) -> TranscriptWriterConfig:
    write_config: TranscriptWriterConfig = TranscriptWriterConfig()
    write_config.timestamp_style = args.format_timestamp
    write_config.probability_style = args.format_probability
    write_config.speaker_style = args.format_speaker
    write_config.language_style = args.format_language
    write_config.verbose = log_level <= logging.INFO
    return write_config


def resolve_speakers(args) -> Optional[int]:
    if args.speakers in (None, "auto"):
        return None
    if args.speakers in ("", 0, "0"):
        return 0
    try:
        return int(args.speakers)
    except (TypeError, ValueError):
        return None


def make_source_config(args, speakers: Optional[int]) -> SourceConfig:
    vttm_path = args.vttm if args.vttm not in (None, "") else None
    return SourceConfig(
        isolate=args.isolate,
        diarize_strategy=resolve_diarize_strategy(args),
        speakers=speakers,
        password=resolve_audio_password(args),
        diarization_file=args.diarization,
        vttm_file=vttm_path,
    )


def preflight_config(*, config: Config, source_config: SourceConfig, args, output_formats: Optional[List[str]] = None) -> bool:
    output_formats = output_formats or build_output_formats(args)
    requested_formats = [fmt for fmt in output_formats if fmt != "stdout"]
    stdout_requested = "stdout" in output_formats or "stdout-nocolor" in output_formats

    if stdout_requested:
        if len(requested_formats) != 1:
            LOG.error("When using -o -, exactly one output format must be selected (e.g. --txt or --jsonl).")
            return False
        if args.quiet:
            LOG.error("Cannot combine -o - with --quiet.")
            return False

    if config.cache is None:
        LOG.error("Artifact cache is not configured. Configure a cache before running.")
        return False
    if source_config.isolate is not None and not config.working_dir:
        LOG.error("Voice isolation requires a working_dir. Provide --workdir or disable --isolate.")
        return False
    return True


def build_output_formats(args, *, default_stdout: bool = True) -> List[str]:
    output_formats: List[str] = []
    if args.ass:
        output_formats.append("ass")
    if args.docx:
        output_formats.append("docx")
    if args.txt:
        output_formats.append("txt")
    if args.md:
        output_formats.append("md")
    if args.jsonl:
        output_formats.append("jsonl")
    if args.json:
        output_formats.append("json")
    if args.outdir == "-":
        output_formats.append("stdout-nocolor" if args.stdout_nocolor else "stdout")
    if default_stdout and not output_formats and args.outdir != "-" and not args.quiet:
        output_formats.append("txt")
    return output_formats


def build_prefixes(config: Config, source_path: str):
    input_name_no_ext = os.path.splitext(os.path.split(source_path)[-1])[0]
    output_prefix_no_ext = os.path.join(config.output_dir, input_name_no_ext)
    if config.working_dir is None:
        working_prefix_no_ext = input_name_no_ext
    else:
        working_prefix_no_ext = os.path.join(config.working_dir, input_name_no_ext)
    return output_prefix_no_ext, working_prefix_no_ext

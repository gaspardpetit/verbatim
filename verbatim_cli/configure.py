"""Helpers to build configs and execution inputs from CLI args."""

import logging
import os
from typing import List, Optional

from verbatim.config import Config
from verbatim_audio.sources.sourceconfig import SourceConfig
from verbatim_files.format.writer import TranscriptWriterConfig

LOG = logging.getLogger(__name__)


def compute_log_level(verbose: int) -> int:
    if verbose <= 1:
        return logging.WARNING
    if verbose == 2:
        return logging.INFO
    return logging.DEBUG


def make_config(args) -> Config:
    config: Config = Config(
        device="cpu" if args.cpu else "auto",
        output_dir=args.outdir,
        working_dir=args.workdir,
        stream=args.stream,
        offline=args.offline,
        model_cache_dir=args.model_cache,
    )
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
        diarize_strategy=args.diarize_policy or args.diarize,
        speakers=speakers,
        diarization_file=args.diarization,
        vttm_file=vttm_path,
    )


def preflight_config(*, config: Config, source_config: SourceConfig, args) -> bool:
    output_formats = build_output_formats(args)
    requested_formats = [fmt for fmt in output_formats if fmt != "stdout"]
    stdout_requested = "stdout" in output_formats

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


def build_output_formats(args) -> List[str]:
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
        output_formats.append("stdout")
    return output_formats


def build_prefixes(config: Config, source_path: str):
    input_name_no_ext = os.path.splitext(os.path.split(source_path)[-1])[0]
    output_prefix_no_ext = os.path.join(config.output_dir, input_name_no_ext)
    if config.working_dir is None:
        working_prefix_no_ext = input_name_no_ext
    else:
        working_prefix_no_ext = os.path.join(config.working_dir, input_name_no_ext)
    return output_prefix_no_ext, working_prefix_no_ext

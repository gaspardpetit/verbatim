"""Helpers to build configs and execution inputs from CLI args."""

import logging
import os
from typing import List, Optional

from verbatim.audio.sources.sourceconfig import SourceConfig
from verbatim.config import Config
from verbatim.transcript.format.writer import TranscriptWriterConfig


def compute_log_level(verbose: int) -> int:
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    return log_levels[min(verbose, len(log_levels) - 1)]


def make_config(args) -> Config:
    config: Config = Config(
        device="cpu" if args.cpu else "auto",
        output_dir=args.outdir,
        working_dir=args.workdir if args.workdir is not None else "",
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


def resolve_diarize(args) -> Optional[int]:
    if args.diarize == "":
        return 0
    if args.diarize is None:
        return None
    return int(args.diarize)


def make_source_config(args, diarize: Optional[int]) -> SourceConfig:
    vttm_path = args.vttm if args.vttm not in (None, "") else None
    return SourceConfig(
        isolate=args.isolate,
        diarize=diarize,
        diarization_file=args.diarization,
        diarization_strategy=args.diarization_strategy,
        vttm_file=vttm_path,
    )


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
    if args.json:
        output_formats.append("json")
    if args.stdout:
        output_formats.append("stdout")
    if args.stdout_nocolor:
        output_formats.append("stdout-nocolor")
    return output_formats


def build_prefixes(config: Config, source_path: str):
    input_name_no_ext = os.path.splitext(os.path.split(source_path)[-1])[0]
    output_prefix_no_ext = os.path.join(config.output_dir, input_name_no_ext)
    working_prefix_no_ext = os.path.join(config.working_dir, input_name_no_ext)
    return output_prefix_no_ext, working_prefix_no_ext

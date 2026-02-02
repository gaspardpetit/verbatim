"""Single-file execution helpers for the CLI."""
# pylint: disable=import-outside-toplevel

import logging
import os
from typing import List, Optional

from verbatim.status_hook import SimpleProgressHook, StatusHook
from verbatim.verbatim import execute
from verbatim_audio.sources.audiosource import AudioSource

LOG = logging.getLogger(__name__)


def build_audio_sources(
    *,
    args,
    config,
    source_config,
    source_path: str,
    working_prefix_no_ext: str,
    output_prefix_no_ext: str,
) -> List[AudioSource]:
    from verbatim_audio.sources.factory import create_audio_sources

    return create_audio_sources(
        source_config=source_config,
        device=config.device,
        cache=config.cache,
        input_source=source_path,
        start_time=args.start_time,
        stop_time=args.stop_time,
        working_prefix_no_ext=working_prefix_no_ext,
        output_prefix_no_ext=output_prefix_no_ext,
        stream=config.stream,
    )


def run_execute(
    *,
    source_path: str,
    config,
    write_config,
    audio_sources: List[AudioSource],
    output_formats,
    output_prefix_no_ext: str,
    working_prefix_no_ext: str,
    eval_file,
    status_hook: Optional[StatusHook] = None,
):
    execute(
        source_path=source_path,
        config=config,
        write_config=write_config,
        audio_sources=audio_sources,
        output_formats=output_formats,
        output_prefix_no_ext=output_prefix_no_ext,
        working_prefix_no_ext=working_prefix_no_ext,
        eval_file=eval_file,
        status_hook=status_hook,
    )


def run_single_input(
    *,
    args,
    log_level: int,
    source_path: str,
    config,
    output_prefix_no_ext: Optional[str] = None,
    working_prefix_no_ext: Optional[str] = None,
    output_formats=None,
    default_stdout: bool = True,
) -> bool:
    from verbatim_cli.configure import (
        build_output_formats,
        build_prefixes,
        make_source_config,
        make_write_config,
        preflight_config,
        resolve_speakers,
    )

    if source_path not in ("-", ">") and not os.path.isfile(source_path):
        LOG.critical("Input audio file not found: %s", source_path)
        return False
    config.read_to_cache(
        input_path=source_path,
        vttm_path=args.vttm,
        rttm_path=args.diarization,
    )

    if output_prefix_no_ext is None or working_prefix_no_ext is None:
        output_prefix_no_ext, working_prefix_no_ext = build_prefixes(config, source_path)

    write_config = make_write_config(args, log_level)
    if output_formats is None:
        output_formats = build_output_formats(args, default_stdout=default_stdout)

    status_hook: Optional[StatusHook] = None
    if default_stdout and not args.quiet and args.outdir != "-":
        status_hook = SimpleProgressHook(config=write_config, with_colours=not args.stdout_nocolor)

    speakers = resolve_speakers(args)
    source_config = make_source_config(args, speakers)
    if not preflight_config(config=config, source_config=source_config, args=args, output_formats=output_formats):
        return False

    audio_sources = build_audio_sources(
        args=args,
        config=config,
        source_config=source_config,
        source_path=source_path,
        working_prefix_no_ext=working_prefix_no_ext,
        output_prefix_no_ext=output_prefix_no_ext,
    )

    run_execute(
        source_path=source_path,
        config=config,
        write_config=write_config,
        audio_sources=audio_sources,
        output_formats=output_formats,
        output_prefix_no_ext=output_prefix_no_ext,
        working_prefix_no_ext=working_prefix_no_ext,
        eval_file=args.eval,
        status_hook=status_hook,
    )
    return True

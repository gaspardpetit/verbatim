import logging
import sys
from typing import List

LOG = logging.getLogger(__name__)


def main():
    # pylint: disable=import-outside-toplevel
    from argparse import Namespace

    from verbatim_cli.args import build_parser
    from verbatim_cli.config_file import load_config_file, merge_args, select_profile
    from verbatim_cli.configure import (
        build_output_formats,
        build_prefixes,
        compute_log_level,
        make_config,
        make_source_config,
        make_write_config,
        resolve_speakers,
    )
    from verbatim_cli.env import load_env_file
    from verbatim_cli.run_single import build_audio_sources, run_execute

    parser = build_parser(prog="verbatim")
    base_defaults: Namespace = parser.parse_args([])
    user_args = parser.parse_args()

    cfg_data = {}
    if getattr(user_args, "config", None):
        cfg_data = load_config_file(user_args.config)

    profile_overrides = select_profile(cfg_data, filename=user_args.input)
    args = merge_args(base_defaults, profile_overrides, user_args)

    log_level = compute_log_level(args.verbose)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # load the values from the .env file, if present
    load_env_file()

    # Check if the version option is specified
    if hasattr(args, "version") and args.version:
        return  # Exit if the version option is specified

    config = make_config(args)

    # Handle install-only mode
    if args.install:
        LOG.info("Installing/prefetching models into cache...")
        from verbatim.prefetch import prefetch

        prefetch(model_cache_dir=args.model_cache, whisper_size=config.whisper_model_size)
        LOG.info("Model prefetch complete.")
        return

    source_path = args.input
    output_prefix_no_ext, working_prefix_no_ext = build_prefixes(config, source_path)

    write_config = make_write_config(args, log_level)

    output_formats = build_output_formats(args)

    speakers = resolve_speakers(args)

    LOG.info(
        "Diarization settings: strategy=%s speakers=%s vttm=%s rttm=%s",
        args.diarize,
        speakers,
        args.vttm,
        args.diarization,
    )

    source_config = make_source_config(args, speakers)

    audio_sources: List = build_audio_sources(
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
    )


if __name__ == "__main__":
    main()

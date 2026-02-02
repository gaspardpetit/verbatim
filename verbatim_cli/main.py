import logging
import sys

LOG = logging.getLogger(__name__)


def main():
    # pylint: disable=import-outside-toplevel
    from argparse import Namespace

    from verbatim.logging_utils import configure_status_logger, get_status_logger
    from verbatim_cli.args import build_parser
    from verbatim_cli.config_file import load_config_file, merge_args, select_profile
    from verbatim_cli.configure import (
        compute_log_level,
        make_config,
        resolve_speakers,
    )
    from verbatim_cli.env import load_env_file
    from verbatim_cli.run_single import run_single_input

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

    log_format = "%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s"
    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    configure_status_logger(verbose=args.verbose, fmt=log_format, datefmt="%Y-%m-%dT%H:%M:%SZ")
    status_log = get_status_logger()

    # load the values from the .env file, if present
    load_env_file()

    # Check if the version option is specified
    if hasattr(args, "version") and args.version:
        return  # Exit if the version option is specified

    config = make_config(args)

    if args.input is None and not args.install:
        parser.print_usage(sys.stderr)
        LOG.error("Missing input path. Provide an audio file, '-' for stdin, or '>' for microphone.")
        return

    # Handle install-only mode
    if args.install:
        status_log.info("Installing/prefetching models into cache...")
        from verbatim.prefetch import prefetch

        prefetch(model_cache_dir=args.model_cache, whisper_size=config.whisper_model_size)
        status_log.info("Model prefetch complete.")
        return

    source_path = args.input
    speakers = resolve_speakers(args)

    status_log.info(
        "Diarization settings: strategy=%s speakers=%s vttm=%s rttm=%s",
        args.diarize,
        speakers,
        args.vttm,
        args.diarization,
    )

    run_single_input(
        args=args,
        log_level=log_level,
        source_path=source_path,
        config=config,
        default_stdout=True,
    )


if __name__ == "__main__":
    main()

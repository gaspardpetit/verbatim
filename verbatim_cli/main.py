import logging
import sys

LOG = logging.getLogger(__name__)


def main():
    # pylint: disable=import-outside-toplevel
    from argparse import Namespace

    from verbatim.logging_utils import configure_status_logger, get_status_logger
    from verbatim_cli.args import build_parser
    from verbatim_cli.config_file import find_legacy_config_keys, load_config_file, merge_args, select_profile
    from verbatim_cli.configure import (
        apply_env_defaults,
        make_config,
        make_source_config,
        resolve_log_level,
        resolve_speakers,
        resolve_status_verbose,
    )
    from verbatim_cli.env import load_env_file
    from verbatim_cli.run_single import run_single_input

    parser = build_parser(prog="verbatim")
    base_defaults: Namespace = parser.parse_args([])
    user_args = parser.parse_args()

    # Load the values from the .env file, if present, before resolving env-backed defaults.
    load_env_file()

    cfg_data = {}
    if getattr(user_args, "config", None):
        cfg_data = load_config_file(user_args.config)

    profile_overrides = select_profile(cfg_data, filename=user_args.input)
    args = merge_args(base_defaults, profile_overrides, user_args)
    args = apply_env_defaults(args, base_defaults)

    if not args.input:
        parser.error("Input audio file must be specified")
    source_path = args.input

    log_level = resolve_log_level(args, base_defaults)
    status_verbose = resolve_status_verbose(args, base_defaults)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = "%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s"
    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%dT%H:%M:%SZ")
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(log_level)
    stderr_handler.setFormatter(formatter)
    handlers = [stderr_handler]
    status_file_handler = None
    if getattr(args, "log_file", None):
        file_handler = logging.FileHandler(args.log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

        status_file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
        status_file_handler.setLevel(logging.DEBUG if status_verbose >= 2 else logging.INFO)
        status_file_handler.setFormatter(formatter)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    configure_status_logger(verbose=status_verbose, fmt=log_format, datefmt="%Y-%m-%dT%H:%M:%SZ", file_handler=status_file_handler)
    status_log = get_status_logger()

    if cfg_data and getattr(user_args, "config", None):
        for path, replacement in find_legacy_config_keys(cfg_data):
            LOG.warning(
                "Config file %s uses deprecated key '%s'; use '%s' instead. This warning shim will be removed in the next minor version bump.",
                user_args.config,
                path,
                replacement,
            )

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

        prefetch(config=config, source_config=make_source_config(args, speakers=resolve_speakers(args)))
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

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable, List

from verbatim_cli.args import add_shared_arguments
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

LOG = logging.getLogger(__name__)


def iter_input_files(batch_dir: Path, patterns: Iterable[str], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(batch_dir.rglob(pattern))
        else:
            files.extend(batch_dir.glob(pattern))
    # Deduplicate and sort for determinism
    return sorted({p.resolve() for p in files if p.is_file()})


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verbatim-batch", description="Batch transcription for Verbatim")
    parser.add_argument("--batch-dir", help="Directory to scan for input audio files")
    parser.add_argument(
        "--match",
        nargs="*",
        default=["*.wav", "*.mp3", "*.m4a", "*.mp4"],
        help="Glob patterns to include (default: *.wav *.mp3 *.m4a *.mp4)",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Glob patterns to ignore",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when searching for input files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files when at least one output artifact already exists")

    add_shared_arguments(parser, include_input=False, prog="verbatim-batch")
    return parser


def outputs_exist(output_prefix_no_ext: str, output_formats: List[str]) -> bool:
    for fmt in output_formats:
        if fmt == "stdout" or fmt == "stdout-nocolor":
            continue
        suffix = f".{fmt}" if fmt != "json" else ".json"
        candidate = Path(f"{output_prefix_no_ext}{suffix}")
        if candidate.exists():
            return True
    return False


def main():
    parser = build_batch_parser()
    base_defaults: Namespace = parser.parse_args([])
    user_args = parser.parse_args()

    cfg_data = {}
    if getattr(user_args, "config", None):
        cfg_data = load_config_file(user_args.config)
        if "match" in cfg_data:
            user_args.match = cfg_data["match"]

    global_profile = select_profile(cfg_data, filename=None)
    args = merge_args(base_defaults, global_profile, user_args)

    log_level = compute_log_level(args.verbose)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    load_env_file()

    # Handle install-only mode
    if args.install:
        LOG.info("Installing/prefetching models into cache...")
        from verbatim.prefetch import prefetch

        # Use default whisper model size from a temporary config
        cfg = make_config(args)
        prefetch(model_cache_dir=args.model_cache, whisper_size=cfg.whisper_model_size)
        LOG.info("Model prefetch complete.")
        return

    if not args.batch_dir:
        parser.error("Batch directory must be specified via --batch-dir or config file")
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    if not batch_dir.exists():
        parser.error(f"Batch directory does not exist: {batch_dir}")

    inputs = iter_input_files(batch_dir, args.match, args.recursive)
    if args.ignore:
        inputs = [p for p in inputs if not any(p.match(pattern) for pattern in args.ignore)]
    if not inputs:
        LOG.info("No input files found under %s with patterns %s", batch_dir, args.match)
        return

    LOG.info("Found %d input files to process", len(inputs))

    for src_path in inputs:
        source_path = str(src_path)

        profile = select_profile(cfg_data, filename=source_path)
        file_args = merge_args(base_defaults, {**global_profile, **profile}, user_args)

        config = make_config(file_args)
        write_config = make_write_config(file_args, log_level)
        speakers = resolve_speakers(file_args)
        source_config = make_source_config(file_args, speakers)
        output_formats = build_output_formats(file_args)

        output_prefix_no_ext, working_prefix_no_ext = build_prefixes(config, source_path)

        if file_args.skip_existing and outputs_exist(output_prefix_no_ext, output_formats):
            LOG.info("Skipping %s (existing outputs found)", source_path)
            continue

        LOG.info("Processing %s", source_path)
        try:
            audio_sources = build_audio_sources(
                args=file_args,
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
                eval_file=file_args.eval,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            LOG.exception("Failed to process %s", source_path)


if __name__ == "__main__":
    main()

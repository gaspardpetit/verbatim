import argparse
import logging
import sys
from argparse import Namespace
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List

# pylint: disable=import-outside-toplevel
from verbatim_cli.args import add_shared_arguments
from verbatim_cli.config_file import find_legacy_config_keys, load_config_file, merge_args, select_profile
from verbatim_cli.configure import (
    apply_env_defaults,
    build_output_formats,
    build_prefixes,
    make_config,
    make_source_config,
    resolve_log_level,
    resolve_speakers,
)
from verbatim_cli.env import load_env_file
from verbatim_cli.run_single import run_single_input

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


def filter_input_files(inputs: Iterable[Path], ignore_patterns: Iterable[str], *, batch_dir: Path) -> List[Path]:
    ignored = [pattern for pattern in ignore_patterns if pattern]
    if not ignored:
        return list(inputs)

    kept: List[Path] = []
    for path in inputs:
        try:
            relative_name = path.relative_to(batch_dir).as_posix()
        except ValueError:
            relative_name = path.as_posix()
        if any(fnmatch(relative_name, pattern) or fnmatch(path.name, pattern) for pattern in ignored):
            continue
        kept.append(path)
    return kept


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
        if fmt in ("stdout", "stdout-nocolor"):
            continue
        suffix = f".{fmt}" if fmt != "json" else ".json"
        candidate = Path(f"{output_prefix_no_ext}{suffix}")
        if candidate.exists():
            return True
    return False


def _install_target_key(config, source_config) -> tuple:
    return (
        config.model_cache_dir,
        config.stream,
        config.transcriber_backend,
        config.whisper_model_size,
        config.qwen_asr_model_size,
        config.qwen_aligner_model_size,
        config.voxtral_model_size,
        config.language_identifier_backend,
        config.mms_lid_model_size,
        config.non_speech_backend,
        config.ast_audio_model_size,
        source_config.diarize_strategy,
        source_config.isolate,
    )


def collect_install_targets(
    *,
    base_defaults: Namespace,
    user_args: Namespace,
    cfg_data: dict,
    global_profile: dict,
    inputs: Iterable[Path],
) -> list[tuple]:
    targets = []
    seen: set[tuple] = set()
    candidate_args = [merge_args(base_defaults, global_profile, user_args)]
    for source_path in inputs:
        profile = select_profile(cfg_data, filename=str(source_path))
        candidate_args.append(merge_args(base_defaults, {**global_profile, **profile}, user_args))

    for candidate in candidate_args:
        resolved_args = apply_env_defaults(candidate, base_defaults)
        speakers = resolve_speakers(resolved_args)
        config = make_config(resolved_args)
        source_config = make_source_config(resolved_args, speakers=speakers)
        key = _install_target_key(config, source_config)
        if key in seen:
            continue
        seen.add(key)
        targets.append((config, source_config))
    return targets


def main():
    parser = build_batch_parser()
    base_defaults: Namespace = parser.parse_args([])
    user_args = parser.parse_args()

    load_env_file()

    cfg_data = {}
    if getattr(user_args, "config", None):
        cfg_data = load_config_file(user_args.config)

    global_profile = select_profile(cfg_data, filename=None)
    args = merge_args(base_defaults, global_profile, user_args)
    args = apply_env_defaults(args, base_defaults)

    log_level = resolve_log_level(args, base_defaults)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    if cfg_data and getattr(user_args, "config", None):
        for path, replacement in find_legacy_config_keys(cfg_data):
            LOG.warning(
                "Config file %s uses deprecated key '%s'; use '%s' instead. This warning shim will be removed in the next minor version bump.",
                user_args.config,
                path,
                replacement,
            )

    if not args.batch_dir:
        parser.error("Batch directory must be specified via --batch-dir or config file")
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    if not batch_dir.exists():
        parser.error(f"Batch directory does not exist: {batch_dir}")

    include_patterns = args.match
    inputs = iter_input_files(batch_dir, include_patterns, args.recursive)
    if args.ignore:
        inputs = filter_input_files(inputs, args.ignore, batch_dir=batch_dir)

    # Handle install-only mode
    if args.install:
        LOG.info("Installing/prefetching models into cache...")
        from verbatim.prefetch import prefetch

        install_targets = collect_install_targets(
            base_defaults=base_defaults,
            user_args=user_args,
            cfg_data=cfg_data,
            global_profile=global_profile,
            inputs=inputs,
        )
        for config, source_config in install_targets:
            prefetch(config=config, source_config=source_config)
        LOG.info("Model prefetch complete for %d effective batch configuration(s).", len(install_targets))
        return
    if not inputs:
        LOG.info("No input files found under %s with patterns %s", batch_dir, include_patterns)
        return

    LOG.info("Found %d input files to process", len(inputs))

    for src_path in inputs:
        source_path = str(src_path)

        profile = select_profile(cfg_data, filename=source_path)
        file_args = merge_args(base_defaults, {**global_profile, **profile}, user_args)
        file_args = apply_env_defaults(file_args, base_defaults)

        config = make_config(file_args)
        output_formats = build_output_formats(file_args, default_stdout=False)

        output_prefix_no_ext, working_prefix_no_ext = build_prefixes(config, source_path)

        if file_args.skip_existing and outputs_exist(output_prefix_no_ext, output_formats):
            LOG.info("Skipping %s (existing outputs found)", source_path)
            continue

        LOG.info("Processing %s", source_path)
        try:
            run_single_input(
                args=file_args,
                log_level=log_level,
                source_path=source_path,
                config=config,
                output_prefix_no_ext=output_prefix_no_ext,
                working_prefix_no_ext=working_prefix_no_ext,
                output_formats=output_formats,
                default_stdout=False,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            LOG.exception("Failed to process %s", source_path)


if __name__ == "__main__":
    main()

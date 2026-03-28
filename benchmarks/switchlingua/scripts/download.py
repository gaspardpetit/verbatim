"""Download SwitchLingua datasets from Hugging Face."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from verbatim_cli.env import load_env_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import infer_language_names

LOG = logging.getLogger(__name__)

DEFAULT_TEXT_REPO = "Shelton1013/SwitchLingua_text"
DEFAULT_AUDIO_REPO = "Shelton1013/SwitchLingua_audio"
DEFAULT_RATE_LIMIT_SLEEP_SECONDS = 60
DEFAULT_RATE_LIMIT_MAX_RETRIES = 10


def _require_token(token: Optional[str]) -> str:
    if token:
        return token
    raise RuntimeError(
        "SwitchLingua datasets are gated. Set HUGGINGFACE_TOKEN or HF_TOKEN (or pass --token) after accepting the dataset terms on Hugging Face."
    )


def _rate_limit_sleep_seconds() -> int:
    raw = os.getenv("SWITCHLINGUA_RATE_LIMIT_SLEEP_SECONDS")
    if raw is None:
        return DEFAULT_RATE_LIMIT_SLEEP_SECONDS
    try:
        value = int(raw)
    except ValueError:
        LOG.warning("Invalid SWITCHLINGUA_RATE_LIMIT_SLEEP_SECONDS=%r; using default %d", raw, DEFAULT_RATE_LIMIT_SLEEP_SECONDS)
        return DEFAULT_RATE_LIMIT_SLEEP_SECONDS
    return max(1, value)


def _rate_limit_max_retries() -> int:
    raw = os.getenv("SWITCHLINGUA_RATE_LIMIT_MAX_RETRIES")
    if raw is None:
        return DEFAULT_RATE_LIMIT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError:
        LOG.warning("Invalid SWITCHLINGUA_RATE_LIMIT_MAX_RETRIES=%r; using default %d", raw, DEFAULT_RATE_LIMIT_MAX_RETRIES)
        return DEFAULT_RATE_LIMIT_MAX_RETRIES
    return max(0, value)


def _extract_retry_after_seconds(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _is_rate_limited(exc: BaseException) -> bool:
    if not isinstance(exc, HfHubHTTPError):
        return False
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 429:
        return True
    return "Too Many Requests" in str(exc)


def _rate_limit_backoff_seconds(
    *,
    base_sleep_seconds: int,
    consecutive_failures: int,
    retry_after_seconds: Optional[int],
) -> int:
    if retry_after_seconds is not None and retry_after_seconds > 0:
        return max(base_sleep_seconds, retry_after_seconds)
    multiplier = 2 ** min(max(consecutive_failures, 0), 2)
    return min(300, max(1, base_sleep_seconds) * multiplier)


def _handle_rate_limit(
    *,
    exc: BaseException,
    base_sleep_seconds: int,
    consecutive_failures: int,
    max_consecutive_failures: int,
    max_workers: int,
) -> int:
    retry_after = _extract_retry_after_seconds(exc)
    consecutive_failures += 1
    wait_seconds = _rate_limit_backoff_seconds(
        base_sleep_seconds=base_sleep_seconds,
        consecutive_failures=consecutive_failures,
        retry_after_seconds=retry_after,
    )

    if consecutive_failures > max_consecutive_failures:
        raise RuntimeError(
            "Hugging Face rate limited the SwitchLingua download. "
            f"Retry with a lower concurrency (current MAX_WORKERS={max_workers}, recommended: 1), "
            "or restrict the sync with ALLOW_PATTERN, for example "
            '`make -C benchmarks/switchlingua install ALLOW_PATTERN="Arabic/*.m4a"`.'
        ) from exc

    LOG.warning(
        "Hugging Face rate limited the SwitchLingua download (HTTP 429). Sleeping %ds and retrying "
        "(consecutive failures=%d/%d). Tip: lower concurrency (current MAX_WORKERS=%d, recommended: 1).",
        wait_seconds,
        consecutive_failures,
        max_consecutive_failures,
        max_workers,
    )
    time.sleep(wait_seconds)
    return consecutive_failures


def download_repo(
    repo_id: str,
    *,
    outdir: Path,
    token: Optional[str],
    max_workers: int,
    allow_patterns: Optional[List[str]],
    ignore_patterns: Optional[List[str]],
) -> Path:
    LOG.info("Downloading %s", repo_id)
    local_dir = outdir / repo_id.split("/")[-1]
    token_value = _require_token(token)
    api = HfApi(token=token_value)
    base_sleep_seconds = _rate_limit_sleep_seconds()
    max_consecutive_failures = _rate_limit_max_retries()
    consecutive_failures = 0

    while True:
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except HfHubHTTPError as exc:
            if _is_rate_limited(exc):
                consecutive_failures = _handle_rate_limit(
                    exc=exc,
                    base_sleep_seconds=base_sleep_seconds,
                    consecutive_failures=consecutive_failures,
                    max_consecutive_failures=max_consecutive_failures,
                    max_workers=max_workers,
                )
                continue
            raise

        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=token_value,
                max_workers=max_workers,
                resume_download=True,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            return local_dir
        except LocalEntryNotFoundError as exc:
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                raise RuntimeError(
                    "The SwitchLingua download did not complete and the requested files were not available in the local cache. "
                    "This commonly happens after an HF rate limit. Re-run the install with MAX_WORKERS=1 or restrict the sync with ALLOW_PATTERN."
                ) from exc
            LOG.warning(
                "SwitchLingua download incomplete (cache miss). Sleeping %ds and retrying (consecutive failures=%d/%d).",
                base_sleep_seconds,
                consecutive_failures,
                max_consecutive_failures,
            )
            time.sleep(base_sleep_seconds)
        except HfHubHTTPError as exc:
            if _is_rate_limited(exc):
                consecutive_failures = _handle_rate_limit(
                    exc=exc,
                    base_sleep_seconds=base_sleep_seconds,
                    consecutive_failures=consecutive_failures,
                    max_consecutive_failures=max_consecutive_failures,
                    max_workers=max_workers,
                )
                continue
            raise


def _list_repo_files(repo_id: str, *, token: Optional[str]) -> None:
    api = HfApi(token=_require_token(token))
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    print(f"{repo_id} files:")
    for filename in files:
        print(f"  - {filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks/switchlingua/scripts/download.py",
        description="Download SwitchLingua datasets from Hugging Face.",
    )
    parser.add_argument("--outdir", default="ext/switchlingua", help="Base directory for dataset downloads")
    parser.add_argument("--text-repo", default=DEFAULT_TEXT_REPO, help="Hugging Face repo for the text dataset")
    parser.add_argument("--audio-repo", default=DEFAULT_AUDIO_REPO, help="Hugging Face repo for the audio dataset")
    parser.add_argument("--token", default=None, help="Hugging Face access token (or use HUGGINGFACE_TOKEN/HF_TOKEN)")
    parser.add_argument("--env", default=".env", help="Optional .env file to load (default: .env)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Max parallel download workers for Hugging Face snapshot (lower avoids rate limits)",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Glob pattern(s) to restrict which files are downloaded (repeatable).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Glob pattern(s) to skip downloading files (repeatable).",
    )
    parser.add_argument("--text-only", action="store_true", help="Download text dataset only")
    parser.add_argument("--audio-only", action="store_true", help="Download audio dataset only")
    parser.add_argument("--list-files", action="store_true", help="List dataset files and exit (requires token)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.env:
        load_env_file(args.env)

    token = args.token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    if args.list_files:
        _list_repo_files(args.audio_repo, token=token)
        _list_repo_files(args.text_repo, token=token)
        return 0

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    allow_patterns = list(args.allow_pattern or [])
    for language in infer_language_names(allow_patterns):
        metadata_name = f"{language}.csv"
        if metadata_name not in allow_patterns:
            allow_patterns.append(metadata_name)

    if not args.audio_only:
        download_repo(
            args.text_repo,
            outdir=outdir,
            token=token,
            max_workers=args.max_workers,
            allow_patterns=allow_patterns or None,
            ignore_patterns=args.ignore_pattern,
        )
    if not args.text_only:
        download_repo(
            args.audio_repo,
            outdir=outdir,
            token=token,
            max_workers=args.max_workers,
            allow_patterns=allow_patterns or None,
            ignore_patterns=args.ignore_pattern,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

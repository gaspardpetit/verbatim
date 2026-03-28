"""Download SwitchLingua datasets from Hugging Face and build a manifest for benchmarking."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from huggingface_hub import HfApi, snapshot_download

from verbatim_cli.env import load_env_file

LOG = logging.getLogger(__name__)

DEFAULT_TEXT_REPO = "Shelton1013/SwitchLingua_text"
DEFAULT_AUDIO_REPO = "Shelton1013/SwitchLingua_audio"

PATH_FIELDS = ("audio", "audio_path", "audio_file", "path", "file", "filename", "file_name")
REFERENCE_FIELDS = ("text", "sentence", "transcript", "reference_text", "data_generation_result", "instances", "utterances", "texts")
ID_FIELDS = ("id", "item_id", "sample_id", "audio_id", "uid", "file_name")
LANGUAGE_FIELDS = ("languages", "language_pair", "first_language", "second_language", "matrix_language", "embedded_language")
METADATA_KEYS = (
    "topic",
    "tense",
    "perspective",
    "cs_ratio",
    "gender",
    "age",
    "education_level",
    "first_language",
    "second_language",
    "conversation_type",
    "cs_function",
    "cs_type",
    "language_pair",
)


@dataclass
class ManifestItem:
    item_id: str
    audio_path: str
    reference_texts: List[str]
    metadata: dict[str, Any]


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _canonicalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key, value in record.items():
        canonical = _canonicalize_key(key)
        if canonical and canonical not in normalized:
            normalized[canonical] = value
    return normalized


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        values: List[str] = []
        for entry in value:
            if isinstance(entry, dict):
                text_value = _first_present(entry, ("text", "utterance", "content", "ref_text"))
                if text_value not in (None, ""):
                    values.append(str(text_value).strip())
            elif entry not in (None, ""):
                values.append(str(entry).strip())
        return [value for value in values if value]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _load_records(path: Path) -> List[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [_canonicalize_record(dict(row)) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [_canonicalize_record(json.loads(line)) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [_canonicalize_record(item) if isinstance(item, dict) else item for item in payload]
        if isinstance(payload, dict):
            for key in ("items", "records", "data", "instances"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [_canonicalize_record(item) if isinstance(item, dict) else item for item in value]
            return [_canonicalize_record(payload)]
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def _find_metadata_file(root: Path) -> Optional[Path]:
    candidates = []
    for name in ("metadata.csv", "metadata.jsonl", "metadata.json"):
        candidate = root / name
        if candidate.exists():
            candidates.append(candidate)
    data_dir = root / "data"
    for name in ("metadata.csv", "metadata.jsonl", "metadata.json"):
        candidate = data_dir / name
        if candidate.exists():
            candidates.append(candidate)

    if candidates:
        return candidates[0]

    root_matches = sorted(root.glob("*.csv")) + sorted(root.glob("*.jsonl")) + sorted(root.glob("*.json"))
    if len(root_matches) == 1:
        return root_matches[0]

    data_matches = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.jsonl")) + sorted(data_dir.glob("*.json"))
    if len(data_matches) == 1:
        return data_matches[0]

    return None


def _infer_metadata_from_patterns(root: Path, patterns: Optional[Sequence[str]]) -> List[Path]:
    if not patterns:
        return []
    language_names: set[str] = set()
    for pattern in patterns:
        normalized = pattern.replace("\\", "/").strip()
        if "/" in normalized:
            head = normalized.split("/", 1)[0].strip()
            if head and head not in (".", "..") and "*" not in head and "?" not in head:
                language_names.add(head)
    candidates = []
    for language in sorted(language_names):
        candidate = root / f"{language}.csv"
        if candidate.exists():
            candidates.append(candidate)
    return candidates


def _infer_language_names(patterns: Optional[Sequence[str]]) -> List[str]:
    if not patterns:
        return []
    language_names: set[str] = set()
    for pattern in patterns:
        normalized = pattern.replace("\\", "/").strip()
        if "/" in normalized:
            head = normalized.split("/", 1)[0].strip()
            if head and head not in (".", "..") and "*" not in head and "?" not in head:
                language_names.add(head)
    return sorted(language_names)


def _resolve_audio_path(value: str, *, audio_root: Path, record_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (record_root, audio_root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (record_root / path).resolve()


def _build_manifest_items(
    records: List[dict[str, Any]],
    *,
    audio_root: Path,
    record_root: Path,
    limit: Optional[int],
    skip_invalid: bool,
) -> List[ManifestItem]:
    items: List[ManifestItem] = []
    for record in records:
        audio_value = _first_present(record, PATH_FIELDS)
        if not audio_value:
            message = f"Record is missing audio field (expected one of {PATH_FIELDS})"
            if skip_invalid:
                LOG.warning("%s. Skipping record.", message)
                continue
            raise ValueError(message)

        text_values: List[str] = []
        for key in REFERENCE_FIELDS:
            if key in record:
                text_values = _string_list(record[key])
                if text_values:
                    break
        if not text_values:
            message = "Record does not provide reference text fields."
            if skip_invalid:
                LOG.warning("%s Skipping record with audio=%s.", message, audio_value)
                continue
            raise ValueError(message)

        item_id = str(_first_present(record, ID_FIELDS) or Path(str(audio_value)).stem)
        audio_path = _resolve_audio_path(str(audio_value), audio_root=audio_root, record_root=record_root)

        metadata = {key: record[key] for key in METADATA_KEYS if key in record and record[key] not in (None, "")}
        for key in LANGUAGE_FIELDS:
            if key in record and key not in metadata and record[key] not in (None, ""):
                metadata[key] = record[key]

        items.append(
            ManifestItem(
                item_id=item_id,
                audio_path=str(audio_path),
                reference_texts=text_values,
                metadata=metadata,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def _write_manifest(items: List[ManifestItem], *, path: Path, audio_root: Optional[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_root = path.parent.resolve()
    audio_root_resolved = audio_root.resolve() if audio_root is not None else None

    def _normalize_audio_path(value: str) -> str:
        audio_path = Path(value).resolve()
        if audio_root_resolved and audio_path.is_relative_to(audio_root_resolved):
            return str(audio_path.relative_to(audio_root_resolved))
        if audio_path.is_relative_to(manifest_root):
            return str(audio_path.relative_to(manifest_root))
        return str(audio_path)

    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            record = {
                "id": item.item_id,
                "audio_path": _normalize_audio_path(item.audio_path),
                "data_generation_result": item.reference_texts,
                **item.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _require_token(token: Optional[str]) -> str:
    if token:
        return token
    raise RuntimeError(
        "SwitchLingua datasets are gated. Set HUGGINGFACE_TOKEN or HF_TOKEN (or pass --token) after accepting the dataset terms on Hugging Face."
    )


def _download_repo(
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
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=_require_token(token),
        max_workers=max_workers,
        resume_download=True,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return local_dir


def _list_repo_files(repo_id: str, *, token: Optional[str]) -> None:
    api = HfApi(token=_require_token(token))
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    print(f"{repo_id} files:")
    for filename in files:
        print(f"  - {filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks/switchlingua/scripts/download.py",
        description="Download SwitchLingua datasets from Hugging Face and build a manifest.",
    )
    parser.add_argument("--outdir", default="ext/switchlingua", help="Base directory for downloads and generated manifest")
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
    parser.add_argument("--manifest", default="ext/switchlingua/manifest.jsonl", help="Path for the generated manifest JSONL")
    parser.add_argument(
        "--metadata",
        action="append",
        default=None,
        help="Explicit metadata file to drive manifest generation (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to include in the manifest")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip records missing audio or text fields")
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

    download_text = not args.audio_only
    download_audio = not args.text_only

    audio_dir = None
    # Enable xet-based downloads if hf_xet is installed.
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")
    allow_patterns = list(args.allow_pattern or [])
    inferred_languages = _infer_language_names(allow_patterns)
    for language in inferred_languages:
        metadata_name = f"{language}.csv"
        if metadata_name not in allow_patterns:
            allow_patterns.append(metadata_name)

    if download_text:
        _download_repo(
            args.text_repo,
            outdir=outdir,
            token=token,
            max_workers=args.max_workers,
            allow_patterns=allow_patterns or None,
            ignore_patterns=args.ignore_pattern,
        )
    if download_audio:
        audio_dir = _download_repo(
            args.audio_repo,
            outdir=outdir,
            token=token,
            max_workers=args.max_workers,
            allow_patterns=allow_patterns or None,
            ignore_patterns=args.ignore_pattern,
        )

    metadata_paths: List[Path] = []
    if args.metadata:
        metadata_paths = [Path(entry).expanduser().resolve() for entry in args.metadata]
    else:
        if audio_dir is None:
            raise RuntimeError("Manifest generation requires audio dataset. Re-run without --text-only or pass --metadata.")
        metadata_paths = _infer_metadata_from_patterns(audio_dir, args.allow_pattern)
        if not metadata_paths:
            candidate = _find_metadata_file(audio_dir)
            if candidate is not None:
                metadata_paths = [candidate]

    if not metadata_paths:
        raise RuntimeError("Unable to locate a metadata file for the audio dataset. Pass --metadata <path> once you identify the metadata file.")

    audio_root = audio_dir if audio_dir is not None else metadata_paths[0].parent
    items: List[ManifestItem] = []
    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            raise RuntimeError(f"Metadata file does not exist: {metadata_path}")
        records = _load_records(metadata_path)
        items.extend(
            _build_manifest_items(
                records,
                audio_root=audio_root,
                record_root=metadata_path.parent,
                limit=None,
                skip_invalid=args.skip_invalid,
            )
        )
        if args.limit is not None and len(items) >= args.limit:
            items = items[: args.limit]
            break
    if not items:
        raise RuntimeError("No valid manifest items were produced. Check the metadata file contents.")

    manifest_path = Path(args.manifest).expanduser().resolve()
    _write_manifest(items, path=manifest_path, audio_root=audio_root)

    LOG.info("Wrote %d manifest items to %s", len(items), manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

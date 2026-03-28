"""Build a benchmark-ready SwitchLingua master manifest."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from verbatim.languages import normalize_language
from verbatim_benchmarks.normalizers import BasicTextNormalizer

LOG = logging.getLogger(__name__)

NORMALIZER = BasicTextNormalizer()
LANGUAGE_FALLBACKS = {
    "arabic": ["ar", "en"],
    "cantonese": ["yue", "en"],
    "french": ["fr", "en"],
    "german": ["de", "en"],
    "hindi": ["hi", "en"],
    "italian": ["it", "en"],
    "japanese": ["ja", "en"],
    "korean": ["ko", "en"],
    "mandarin": ["zh", "en"],
    "russian": ["ru", "en"],
    "spanish": ["es", "en"],
}
LANGUAGE_AUDIO_DIRS = {
    key: key.title() for key in LANGUAGE_FALLBACKS
}

PATH_FIELDS = ("audio", "audio_path", "audio_file", "path", "file", "filename", "file_name")
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
    text: str
    normalized: str
    languages: List[str]
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
    if isinstance(value, str):
        stripped = value.strip()
        parsed_values = _extract_embedded_text_list(stripped)
        if parsed_values is not None:
            return parsed_values
        return [stripped] if stripped else []
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
    return []


def _extract_embedded_text_list(value: str) -> Optional[List[str]]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate or candidate[0] not in "[{":
        return None

    parsed = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
            break
        except Exception:
            continue
    if parsed is None:
        return None

    if isinstance(parsed, dict):
        for key in ("instances", "data_generation_result", "texts", "utterances"):
            extracted = _string_list(parsed.get(key))
            if extracted:
                return extracted
        text_value = _first_present(parsed, ("text", "turn", "utterance", "content", "ref_text"))
        extracted = _string_list(text_value)
        return extracted or None

    if isinstance(parsed, list):
        extracted = _string_list(parsed)
        return extracted or None

    return None


def _normalize_language(value: Any) -> Optional[str]:
    return normalize_language(value)


def _parse_languages(record: dict[str, Any], dataset_lang: Optional[str]) -> List[str]:
    languages: List[str] = []
    raw = record.get("languages")
    if isinstance(raw, str):
        for part in raw.replace("/", ",").split(","):
            lang = _normalize_language(part)
            if lang and lang not in languages:
                languages.append(lang)
    for key in ("first_language", "second_language", "matrix_language", "embedded_language"):
        lang = _normalize_language(record.get(key))
        if lang and lang not in languages:
            languages.append(lang)
    if not languages and dataset_lang and dataset_lang in LANGUAGE_FALLBACKS:
        languages = list(LANGUAGE_FALLBACKS[dataset_lang])
    return languages


def _conversation_key(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_", 1)[0]


def _turn_index(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    match = re.search(r"_(\d+)$", stem)
    if not match:
        return None
    return int(match.group(1))


def _extract_text(record: dict[str, Any], filename: str, conversation_size: int) -> str:
    candidates = []
    for key in ("text", "data_generation_result", "reference_text", "transcript"):
        if key in record:
            candidates.extend(_string_list(record[key]))
            if candidates:
                break
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    if conversation_size <= 1:
        return " ".join(candidates)
    turn_index = _turn_index(filename)
    if turn_index is not None and 0 <= turn_index < len(candidates):
        return candidates[turn_index]
    return candidates[0]


def dataset_lang_from_metadata_path(path: Path) -> Optional[str]:
    normalized = path.stem.strip().lower()
    if normalized.endswith("_eng"):
        normalized = normalized[: -len("_eng")]
    if normalized in LANGUAGE_FALLBACKS:
        return normalized
    return None


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


def find_metadata_files(root: Path) -> List[Path]:
    candidates: List[Path] = []
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
        return candidates

    root_matches = sorted(root.glob("*.csv")) + sorted(root.glob("*.jsonl")) + sorted(root.glob("*.json"))
    if root_matches:
        return root_matches

    data_matches = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.jsonl")) + sorted(data_dir.glob("*.json"))
    if data_matches:
        return data_matches

    return []


def infer_metadata_from_patterns(root: Path, patterns: Optional[Sequence[str]]) -> List[Path]:
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


def infer_language_names(patterns: Optional[Sequence[str]]) -> List[str]:
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


def _resolve_audio_path(value: str, *, audio_root: Path, record_root: Path, dataset_lang: Optional[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (record_root, audio_root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    if dataset_lang and len(path.parts) == 1:
        dataset_dir = LANGUAGE_AUDIO_DIRS.get(dataset_lang.strip().lower())
        if dataset_dir:
            candidate = (audio_root / dataset_dir / path).resolve()
            if candidate.exists():
                return candidate
    return (record_root / path).resolve()


def build_manifest_items(
    records: List[dict[str, Any]],
    *,
    audio_root: Path,
    record_root: Path,
    dataset_lang: Optional[str],
    limit: Optional[int],
    skip_invalid: bool,
) -> List[ManifestItem]:
    items: List[ManifestItem] = []
    conversation_sizes: dict[str, int] = {}
    for record in records:
        filename = str(_first_present(record, PATH_FIELDS) or "").strip()
        if not filename:
            continue
        key = _conversation_key(filename)
        conversation_sizes[key] = conversation_sizes.get(key, 0) + 1

    for record in records:
        audio_value = _first_present(record, PATH_FIELDS)
        if not audio_value:
            message = f"Record is missing audio field (expected one of {PATH_FIELDS})"
            if skip_invalid:
                LOG.warning("%s. Skipping record.", message)
                continue
            raise ValueError(message)

        filename = str(audio_value).strip()
        text = _extract_text(record, filename, conversation_sizes.get(_conversation_key(filename), 1))
        if not text:
            message = "Record does not provide reference text fields."
            if skip_invalid:
                LOG.warning("%s Skipping record with audio=%s.", message, audio_value)
                continue
            raise ValueError(message)

        item_id = str(_first_present(record, ID_FIELDS) or Path(str(audio_value)).stem)
        audio_path = _resolve_audio_path(str(audio_value), audio_root=audio_root, record_root=record_root, dataset_lang=dataset_lang)

        metadata = {key: record[key] for key in METADATA_KEYS if key in record and record[key] not in (None, "")}
        for key in LANGUAGE_FIELDS:
            if key in record and key not in metadata and record[key] not in (None, ""):
                metadata[key] = record[key]
        languages = _parse_languages(record, dataset_lang)
        normalized = NORMALIZER(text).strip()
        if dataset_lang:
            metadata.setdefault("set", dataset_lang)

        items.append(
            ManifestItem(
                item_id=item_id,
                audio_path=str(audio_path),
                text=text,
                normalized=normalized,
                languages=languages,
                metadata=metadata,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def write_manifest(items: List[ManifestItem], *, path: Path, audio_root: Optional[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_root = path.parent.resolve()
    audio_root_resolved = audio_root.resolve() if audio_root is not None else None

    def _normalize_audio_path(value: str) -> str:
        audio_path = Path(value).resolve()
        if audio_root_resolved and audio_path.is_relative_to(audio_root_resolved):
            return audio_path.relative_to(audio_root_resolved).as_posix()
        if audio_path.is_relative_to(manifest_root):
            return audio_path.relative_to(manifest_root).as_posix()
        return audio_path.as_posix()

    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            record = {
                "id": item.item_id,
                "audio_path": _normalize_audio_path(item.audio_path),
                "text": item.text,
                "normalized": item.normalized,
                "languages": ",".join(item.languages),
                **item.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def build_master_manifest(
    *,
    audio_root: Path,
    manifest_path: Path,
    metadata_paths: Optional[List[Path]],
    allow_patterns: Optional[Sequence[str]],
    limit: Optional[int],
    skip_invalid: bool,
) -> Path:
    resolved_metadata = [Path(entry).expanduser().resolve() for entry in metadata_paths] if metadata_paths else []
    if not resolved_metadata:
        resolved_metadata = infer_metadata_from_patterns(audio_root, allow_patterns)
    if not resolved_metadata:
        resolved_metadata = find_metadata_files(audio_root)
    if not resolved_metadata:
        raise RuntimeError("Unable to locate a metadata file for the audio dataset. Pass --metadata <path> once you identify the metadata file.")

    items: List[ManifestItem] = []
    for metadata_path in resolved_metadata:
        if not metadata_path.exists():
            raise RuntimeError(f"Metadata file does not exist: {metadata_path}")
        records = _load_records(metadata_path)
        items.extend(
            build_manifest_items(
                records,
                audio_root=audio_root,
                record_root=metadata_path.parent,
                dataset_lang=dataset_lang_from_metadata_path(metadata_path),
                limit=None,
                skip_invalid=skip_invalid,
            )
        )
        if limit is not None and len(items) >= limit:
            items = items[: limit]
            break
    if not items:
        raise RuntimeError("No valid manifest items were produced. Check the metadata file contents.")

    resolved_manifest_path = manifest_path.expanduser().resolve()
    write_manifest(items, path=resolved_manifest_path, audio_root=audio_root)
    LOG.info("Wrote %d manifest items to %s", len(items), resolved_manifest_path)
    return resolved_manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks/switchlingua/scripts/manifest.py",
        description="Build a benchmark-ready SwitchLingua master manifest.",
    )
    parser.add_argument("--audio-root", default="ext/switchlingua/SwitchLingua_audio", help="Directory containing the downloaded audio dataset")
    parser.add_argument(
        "--manifest",
        default="benchmarks/switchlingua/manifests/manifest_bootstrap.jsonl",
        help="Path for the generated master manifest",
    )
    parser.add_argument("--metadata", action="append", default=None, help="Explicit metadata file to drive manifest generation (repeatable).")
    parser.add_argument("--allow-pattern", action="append", default=None, help="Subset glob pattern(s) used to infer matching metadata files.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to include in the manifest")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip records missing audio or text fields")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_master_manifest(
        audio_root=Path(args.audio_root).expanduser().resolve(),
        manifest_path=Path(args.manifest).expanduser().resolve(),
        metadata_paths=args.metadata,
        allow_patterns=args.allow_pattern,
        limit=args.limit,
        skip_invalid=args.skip_invalid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

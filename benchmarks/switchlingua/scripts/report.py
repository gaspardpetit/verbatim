"""Aggregate corpus-level WER/CER by system and language from SwitchLingua outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from verbatim.eval.diarization_metrics import compute_utterance_metrics
from verbatim.eval.diarization_utils import normalize_text
from verbatim.languages import normalize_language
from verbatim_benchmarks.normalizers import BasicTextNormalizer

_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCHMARK_ROOT = _SCRIPT_DIR.parent
_DEFAULT_MANIFEST = _BENCHMARK_ROOT / "manifests" / "manifest_bootstrap.jsonl"
_DEFAULT_MANIFEST_ROOT = _BENCHMARK_ROOT / "manifests"
_DEFAULT_OUTDIR = _BENCHMARK_ROOT / "out"

PATH_FIELDS = ("audio", "audio_path", "audio_file", "path", "file", "filename")
ID_FIELDS = ("id", "item", "item_id", "sample_id", "audio_id", "uid")
TEXT_LIST_FIELDS = ("data_generation_result", "instances", "reference_texts", "utterances", "texts")
TEXT_FIELDS = ("text", "transcript", "reference_text", "sentence")
NONLEXICAL_UTTERANCE_RE = re.compile(r"^\[[^\]]+\](?:\s+\[[^\]]+\])*$")
_BASIC_NORMALIZER = BasicTextNormalizer()


@dataclass
class RefItem:
    item_id: str
    ref_text: str
    normalized_text: Optional[str]
    language: str


def _slugify(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value.strip())
    return sanitized.strip("_") or "item"


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _normalize_language(value: Any) -> Optional[str]:
    return normalize_language(value)


def _parse_language(record: dict[str, Any]) -> str:
    languages_value = record.get("languages")
    if languages_value:
        if isinstance(languages_value, str):
            first = languages_value.split(",", 1)[0].strip()
            normalized = _normalize_language(first)
            if normalized:
                return normalized
        elif isinstance(languages_value, list) and languages_value:
            normalized = _normalize_language(languages_value[0])
            if normalized:
                return normalized
    for key in ("matrix_language", "first_language", "language", "lang", "language_pair"):
        value = record.get(key)
        if value:
            if isinstance(value, str) and "-" in value:
                value = value.split("-")[0]
            normalized = _normalize_language(value)
            if normalized:
                return normalized
    set_value = record.get("set")
    if set_value:
        normalized = _normalize_language(set_value)
        if normalized:
            return normalized
    return "und"


def _normalize_form_text(text: str, lang: str) -> str:
    return _BASIC_NORMALIZER(text or "").strip()


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        stripped = value.strip()
        parsed_values = _extract_embedded_text_list(stripped)
        if parsed_values is not None:
            return parsed_values
    if isinstance(value, list):
        values = []
        for entry in value:
            if isinstance(entry, dict):
                text_value = _first_present(entry, ("text", "utterance", "content", "ref_text"))
                if text_value not in (None, ""):
                    values.append(str(text_value).strip())
            elif isinstance(entry, str):
                parsed_values = _extract_embedded_text_list(entry.strip())
                if parsed_values is not None:
                    values.extend(parsed_values)
                elif entry not in (None, ""):
                    values.append(str(entry).strip())
            elif entry not in (None, ""):
                values.append(str(entry).strip())
        return [value for value in values if value]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
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


def _reference_from_record(record: dict[str, Any]) -> str:
    text_values: List[str] = []
    for key in TEXT_LIST_FIELDS:
        if key in record:
            text_values = _string_list(record[key])
            if text_values:
                break
    if not text_values:
        for key in TEXT_FIELDS:
            if key in record:
                text_values = _string_list(record[key])
                if text_values:
                    break
    return " ".join(text_values)


def _load_records(path: Path) -> List[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("items", "records", "data", "instances"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def load_manifest(path: Path, *, run_name: Optional[str] = None) -> Dict[str, RefItem]:
    records = _load_records(path)
    ref_map: Dict[str, RefItem] = {}
    normalized_run_name = run_name.strip().lower() if run_name else None
    for record in records:
        if normalized_run_name:
            record_set = str(record.get("set", "")).strip().lower()
            if record_set and record_set != normalized_run_name:
                continue
        audio_value = _first_present(record, PATH_FIELDS)
        item_id = str(_first_present(record, ID_FIELDS) or Path(str(audio_value)).stem)
        normalized_id = _slugify(item_id)
        ref_text = _reference_from_record(record)
        normalized_text = str(record.get("normalized", "")).strip() or None
        language = _parse_language(record)
        ref_map[normalized_id] = RefItem(item_id=normalized_id, ref_text=ref_text, normalized_text=normalized_text, language=language)
    return ref_map


def _iter_system_dirs(outdir: Path) -> Iterable[Tuple[Optional[str], str, Path]]:
    for entry in outdir.iterdir():
        if not entry.is_dir():
            continue
        child_dirs = [d for d in entry.iterdir() if d.is_dir() and not d.name.endswith("__partial")]
        # Detect run-name folder: child directories that themselves contain item subdirs with jsons.
        is_run_folder = False
        for child in child_dirs:
            grandchild_dirs = [g for g in child.iterdir() if g.is_dir() and not g.name.endswith("__partial")]
            if any(list(g.glob("*.json")) for g in grandchild_dirs):
                is_run_folder = True
                break
        if is_run_folder:
            for system_dir in child_dirs:
                yield entry.name, system_dir.name, system_dir
        else:
            yield None, entry.name, entry


def _read_hyp_text(output_json: Path) -> Optional[str]:
    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    utts = data.get("utterances", [])
    filtered = []
    for utt in utts:
        text = utt.get("text", "")
        if not text:
            continue
        stripped = text.strip()
        if not stripped:
            continue
        if (not utt.get("words")) and NONLEXICAL_UTTERANCE_RE.fullmatch(stripped):
            continue
        filtered.append(stripped)
    return " ".join(filtered)


def _infer_manifest_for_run(run_name: str, manifest_root: Path) -> Optional[Path]:
    benchmark_candidate = manifest_root / f"benchmark_manifest_{run_name}.csv"
    if benchmark_candidate.exists():
        return benchmark_candidate
    candidate = manifest_root / f"manifest_{run_name}.jsonl"
    if candidate.exists():
        return candidate
    return None


def _iter_output_jsons(system_dir: Path) -> Iterable[Path]:
    def _ignore_walk_error(_: OSError) -> None:
        return

    for root, dirnames, filenames in os.walk(system_dir, topdown=True, onerror=_ignore_walk_error):
        dirnames[:] = [dirname for dirname in dirnames if not dirname.endswith("__partial")]
        for filename in filenames:
            if not filename.endswith(".json") or filename == "summary.json":
                continue
            yield Path(root) / filename


def compute_report(manifest: Optional[Path], outdir: Path, manifest_root: Path) -> List[dict[str, Any]]:
    manifest_cache: Dict[str, Dict[str, RefItem]] = {}
    results: Dict[Tuple[str, str], dict[str, Any]] = {}

    for run_name, system_name, system_dir in _iter_system_dirs(outdir):
        manifest_key = "__explicit__"
        if manifest is None:
            if not run_name:
                continue
            inferred_manifest = _infer_manifest_for_run(run_name, manifest_root)
            if inferred_manifest is None:
                continue
            manifest_key = run_name
            if manifest_key not in manifest_cache:
                manifest_cache[manifest_key] = load_manifest(inferred_manifest, run_name=run_name)
        else:
            manifest_key = run_name or "__explicit__"
            if manifest_key not in manifest_cache:
                manifest_cache[manifest_key] = load_manifest(manifest, run_name=run_name)
        ref_map = manifest_cache[manifest_key]

        if not system_dir.exists():
            continue
        for output_json in _iter_output_jsons(system_dir):
            item_id = output_json.parent.name
            ref_item = ref_map.get(item_id)
            if ref_item is None:
                continue
            hyp_text = _read_hyp_text(output_json)
            if hyp_text is None:
                continue

            key = (run_name or ref_item.language, system_name)
            row = results.setdefault(
                key,
                {
                    "system": system_name,
                    "run": run_name or "",
                    "language": ref_item.language,
                    "items": 0,
                    "wer_sum": 0.0,
                    "wer_count": 0,
                    "cer_sum": 0.0,
                    "cer_count": 0,
                    "wer_sub": 0,
                    "wer_del": 0,
                    "wer_ins": 0,
                    "wer_total": 0,
                    "cer_dist": 0,
                    "cer_total": 0,
                    "sem_sum": 0.0,
                    "sem_count": 0,
                    "saer_sum": 0.0,
                    "saer_count": 0,
                },
            )

            normalized_hyp_text = _normalize_form_text(hyp_text, ref_item.language)
            normalized_ref_text = ref_item.normalized_text or _normalize_form_text(ref_item.ref_text, ref_item.language)
            metrics = compute_utterance_metrics(hyp_text=normalized_hyp_text, ref_text=normalized_ref_text)
            row["items"] += 1
            if metrics.wer_total:
                row["wer_sum"] += (metrics.wer_sub + metrics.wer_delete + metrics.wer_insert) / metrics.wer_total
                row["wer_count"] += 1
            row["wer_sub"] += metrics.wer_sub
            row["wer_del"] += metrics.wer_delete
            row["wer_ins"] += metrics.wer_insert
            row["wer_total"] += metrics.wer_total

            # Optional semantic metrics from score payload.
            try:
                data = json.loads(output_json.read_text(encoding="utf-8"))
                score = data.get("score") or {}
                sem = score.get("sem")
                saer = score.get("saer")
                if sem is not None:
                    row["sem_sum"] += float(sem)
                    row["sem_count"] += 1
                if saer is not None:
                    row["saer_sum"] += float(saer)
                    row["saer_count"] += 1
            except Exception:
                pass

            ref_norm = normalize_text(normalized_ref_text)
            hyp_norm = normalize_text(normalized_hyp_text)
            if ref_norm:
                # Simple edit distance for CER
                a = list(hyp_norm)
                b = list(ref_norm)
                prev = list(range(len(b) + 1))
                for i, ca in enumerate(a, start=1):
                    cur = [i]
                    for j, cb in enumerate(b, start=1):
                        cost = 0 if ca == cb else 1
                        cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
                    prev = cur
                row["cer_dist"] += prev[-1]
                row["cer_total"] += len(b)
                row["cer_sum"] += prev[-1] / len(b)
                row["cer_count"] += 1

    report_rows = []
    for row in results.values():
        wer_total = row["wer_total"]
        cer_total = row["cer_total"]
        corpus_wer = (row["wer_sub"] + row["wer_del"] + row["wer_ins"]) / wer_total if wer_total else None
        corpus_cer = row["cer_dist"] / cer_total if cer_total else None
        report_rows.append(
            {
                "system": row["system"],
                "run": row["run"],
                "language": row["language"],
                "items": row["items"],
                "corpus_wer": corpus_wer,
                "corpus_cer": corpus_cer,
                "total_words": wer_total,
                "total_chars": cer_total,
                "mean_wer": (row["wer_sum"] / row["wer_count"]) if row["wer_count"] else None,
                "mean_cer": (row["cer_sum"] / row["cer_count"]) if row["cer_count"] else None,
                "mean_sem": (row["sem_sum"] / row["sem_count"]) if row["sem_count"] else None,
                "mean_saer": (row["saer_sum"] / row["saer_count"]) if row["saer_count"] else None,
                "wer_sum": row["wer_sum"],
                "wer_count": row["wer_count"],
                "cer_sum": row["cer_sum"],
                "cer_count": row["cer_count"],
                "wer_sub": row["wer_sub"],
                "wer_del": row["wer_del"],
                "wer_ins": row["wer_ins"],
                "wer_total": row["wer_total"],
                "cer_dist": row["cer_dist"],
                "cer_total": row["cer_total"],
            }
        )

    report_rows.sort(key=lambda r: (r["run"], r["system"]))
    return report_rows


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = []
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def print_table(rows: List[dict[str, Any]]) -> None:
    headers = ["run", "system", "language", "items", "corpus_wer", "corpus_cer", "mean_sem", "mean_saer", "total_words", "total_chars"]
    table_rows: List[List[str]] = []
    for row in rows:

        def fmt(value):
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value) if value is not None else ""

        table_rows.append([fmt(row[h]) for h in headers])
    print(_format_table(headers, table_rows))


def print_summary(rows: List[dict[str, Any]], *, average: bool = False) -> None:
    # Aggregate by run + system across languages.
    grouped: Dict[Tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["run"], row["system"])
        agg = grouped.setdefault(
            key,
            {
                "run": row["run"],
                "system": row["system"],
                "items": 0,
                "wer_sum": 0.0,
                "wer_count": 0,
                "cer_sum": 0.0,
                "cer_count": 0,
                "wer_sub": 0,
                "wer_del": 0,
                "wer_ins": 0,
                "wer_total": 0,
                "cer_dist": 0,
                "cer_total": 0,
                "sem_sum": 0.0,
                "sem_count": 0,
                "saer_sum": 0.0,
                "saer_count": 0,
            },
        )
        agg["wer_sub"] += row.get("wer_sub", 0) or 0
        agg["wer_del"] += row.get("wer_del", 0) or 0
        agg["wer_ins"] += row.get("wer_ins", 0) or 0
        agg["wer_total"] += row.get("wer_total", 0) or 0
        agg["cer_dist"] += row.get("cer_dist", 0) or 0
        agg["cer_total"] += row.get("cer_total", 0) or 0
        agg["wer_sum"] += row.get("wer_sum", 0.0) or 0.0
        agg["wer_count"] += row.get("wer_count", 0) or 0
        agg["cer_sum"] += row.get("cer_sum", 0.0) or 0.0
        agg["cer_count"] += row.get("cer_count", 0) or 0
        agg["items"] += row["items"]
        if row.get("mean_sem") is not None:
            agg["sem_sum"] += row["mean_sem"] * row["items"]
            agg["sem_count"] += row["items"]
        if row.get("mean_saer") is not None:
            agg["saer_sum"] += row["mean_saer"] * row["items"]
            agg["saer_count"] += row["items"]

    headers = ["run", "system", "items", "corpus_wer", "corpus_cer", "mean_sem", "mean_saer", "total_words", "total_chars"]
    table_rows: List[List[str]] = []
    for agg in grouped.values():
        corpus_wer = ((agg["wer_sub"] + agg["wer_del"] + agg["wer_ins"]) / agg["wer_total"]) if agg["wer_total"] else None
        corpus_cer = (agg["cer_dist"] / agg["cer_total"]) if agg["cer_total"] else None
        mean_wer = (agg["wer_sum"] / agg["wer_count"]) if agg["wer_count"] else None
        mean_cer = (agg["cer_sum"] / agg["cer_count"]) if agg["cer_count"] else None
        mean_sem = (agg["sem_sum"] / agg["sem_count"]) if agg["sem_count"] else None
        mean_saer = (agg["saer_sum"] / agg["saer_count"]) if agg["saer_count"] else None
        row = {
            "run": agg["run"],
            "system": agg["system"],
            "items": agg["items"],
            "corpus_wer": mean_wer if average else corpus_wer,
            "corpus_cer": mean_cer if average else corpus_cer,
            "mean_sem": mean_sem,
            "mean_saer": mean_saer,
            "total_words": agg["wer_total"],
            "total_chars": agg["cer_total"],
        }

        def fmt(value):
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value) if value is not None else ""

        table_rows.append([fmt(row[h]) for h in headers])
    print(_format_table(headers, table_rows))


def _pivot_columns(rows: List[dict[str, Any]]) -> tuple[bool, List[str]]:
    use_runs = any(row.get("run") for row in rows)
    columns = sorted({(row["run"] if use_runs else row["language"]) for row in rows})
    return use_runs, columns


def print_pivot(rows: List[dict[str, Any]], *, pair: bool, average: bool = False) -> None:
    use_runs, columns = _pivot_columns(rows)
    grouped: Dict[str, Dict[str, dict[str, Any]]] = {}
    totals: Dict[str, dict[str, Any]] = {}

    for row in rows:
        system = row["system"]
        column_key = row["run"] if use_runs else row["language"]
        grouped.setdefault(system, {})[column_key] = row
        agg = totals.setdefault(
            system,
            {
                "wer_sum": 0.0,
                "wer_count": 0,
                "cer_sum": 0.0,
                "cer_count": 0,
                "wer_sub": 0,
                "wer_del": 0,
                "wer_ins": 0,
                "wer_total": 0,
                "cer_dist": 0,
                "cer_total": 0,
            },
        )
        agg["wer_sub"] += row.get("wer_sub", 0) or 0
        agg["wer_del"] += row.get("wer_del", 0) or 0
        agg["wer_ins"] += row.get("wer_ins", 0) or 0
        agg["wer_total"] += row.get("wer_total", 0) or 0
        agg["cer_dist"] += row.get("cer_dist", 0) or 0
        agg["cer_total"] += row.get("cer_total", 0) or 0
        agg["wer_sum"] += row.get("wer_sum", 0.0) or 0.0
        agg["wer_count"] += row.get("wer_count", 0) or 0
        agg["cer_sum"] += row.get("cer_sum", 0.0) or 0.0
        agg["cer_count"] += row.get("cer_count", 0) or 0

    headers = ["system"] + columns + ["total"]
    table_rows: List[List[str]] = []

    for system in sorted(grouped.keys()):
        row_cells = [system]
        for column_key in columns:
            cell = ""
            if column_key in grouped[system]:
                item = grouped[system][column_key]
                wer = item.get("mean_wer") if average else item.get("corpus_wer")
                cer = item.get("mean_cer") if average else item.get("corpus_cer")
                if wer is not None and cer is not None:
                    cell = f"{cer:.4f}/{wer:.4f}" if pair else f"{wer:.4f}"
            row_cells.append(cell)

        agg = totals.get(system, {})
        total_wer = None
        total_cer = None
        if average:
            if agg.get("wer_count"):
                total_wer = agg["wer_sum"] / agg["wer_count"]
            if agg.get("cer_count"):
                total_cer = agg["cer_sum"] / agg["cer_count"]
        else:
            if agg.get("wer_total"):
                total_wer = (agg["wer_sub"] + agg["wer_del"] + agg["wer_ins"]) / agg["wer_total"]
            if agg.get("cer_total"):
                total_cer = agg["cer_dist"] / agg["cer_total"]
        if total_wer is not None and total_cer is not None:
            total_cell = f"{total_cer:.4f}/{total_wer:.4f}" if pair else f"{total_wer:.4f}"
        else:
            total_cell = ""
        row_cells.append(total_cell)
        table_rows.append([str(cell) for cell in row_cells])
    print(_format_table(headers, table_rows))


def print_semantic_pivot(rows: List[dict[str, Any]]) -> None:
    use_runs, columns = _pivot_columns(rows)
    grouped: Dict[str, Dict[str, dict[str, Any]]] = {}
    totals: Dict[str, dict[str, Any]] = {}

    for row in rows:
        system = row["system"]
        column_key = row["run"] if use_runs else row["language"]
        grouped.setdefault(system, {})[column_key] = row
        agg = totals.setdefault(
            system,
            {
                "sem_sum": 0.0,
                "sem_count": 0,
                "saer_sum": 0.0,
                "saer_count": 0,
            },
        )
        if row.get("mean_sem") is not None:
            agg["sem_sum"] += float(row["mean_sem"]) * int(row["items"])
            agg["sem_count"] += int(row["items"])
        if row.get("mean_saer") is not None:
            agg["saer_sum"] += float(row["mean_saer"]) * int(row["items"])
            agg["saer_count"] += int(row["items"])

    headers = ["system"] + columns + ["total"]
    table_rows: List[List[str]] = []
    has_any_values = False

    for system in sorted(grouped.keys()):
        row_cells = [system]
        system_has_values = False
        for column_key in columns:
            cell = ""
            if column_key in grouped[system]:
                item = grouped[system][column_key]
                sem = item.get("mean_sem")
                saer = item.get("mean_saer")
                if sem is not None and saer is not None:
                    cell = f"{sem:.4f}/{saer:.4f}"
                    system_has_values = True
                    has_any_values = True
            row_cells.append(cell)

        agg = totals.get(system, {})
        total_sem = (agg["sem_sum"] / agg["sem_count"]) if agg.get("sem_count") else None
        total_saer = (agg["saer_sum"] / agg["saer_count"]) if agg.get("saer_count") else None
        if total_sem is not None and total_saer is not None:
            total_cell = f"{total_sem:.4f}/{total_saer:.4f}"
            system_has_values = True
            has_any_values = True
        else:
            total_cell = ""
        row_cells.append(total_cell)

        if system_has_values:
            table_rows.append([str(cell) for cell in row_cells])

    if has_any_values:
        print()
        print("SEM/SAER")
        print(_format_table(headers, table_rows))


def print_items_pivot(rows: List[dict[str, Any]]) -> None:
    use_runs, columns = _pivot_columns(rows)
    grouped: Dict[str, Dict[str, dict[str, Any]]] = {}
    totals: Dict[str, int] = {}

    for row in rows:
        system = row["system"]
        column_key = row["run"] if use_runs else row["language"]
        grouped.setdefault(system, {})[column_key] = row
        totals[system] = totals.get(system, 0) + int(row.get("items", 0) or 0)

    headers = ["system"] + columns + ["total"]
    table_rows: List[List[str]] = []
    for system in sorted(grouped.keys()):
        row_cells = [system]
        for column_key in columns:
            cell = ""
            if column_key in grouped[system]:
                cell = str(int(grouped[system][column_key].get("items", 0) or 0))
            row_cells.append(cell)
        row_cells.append(str(totals.get(system, 0)))
        table_rows.append(row_cells)

    print()
    print("Items")
    print(_format_table(headers, table_rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks/switchlingua/scripts/report.py",
        description="Aggregate corpus WER/CER by system and language.",
    )
    parser.add_argument("--manifest", default=str(_DEFAULT_MANIFEST), help="Manifest JSON/JSONL/CSV for benchmark reporting")
    parser.add_argument("--manifest-root", default=str(_DEFAULT_MANIFEST_ROOT), help="Directory containing legacy manifest_<run>.jsonl files")
    parser.add_argument("--outdir", default=str(_DEFAULT_OUTDIR), help="Output directory containing benchmark runs")
    parser.add_argument("--summary", action="store_true", help="Print summary table aggregated by run+system")
    parser.add_argument("--pivot", action="store_true", help="Print pivot table by run+system with language columns")
    parser.add_argument("--pair", action="store_true", help="When used with --pivot, print CER/WER pairs")
    parser.add_argument("--average", action="store_true", help="Use mean per-item WER/CER instead of corpus-level aggregation")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    rows = compute_report(manifest, outdir, manifest_root)
    if not rows:
        print("No results found.")
        return 1
    if not args.summary and not args.pivot:
        args.pivot = True
        args.pair = True
    if args.pivot:
        print_pivot(rows, pair=args.pair, average=args.average)
        print_semantic_pivot(rows)
        print_items_pivot(rows)
    elif args.summary:
        print_summary(rows, average=args.average)
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
